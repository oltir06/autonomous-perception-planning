"""PyTorch Dataset for CARLA segmentation data.

Provides a Dataset class that loads synchronized RGB + segmentation frame
pairs collected from CARLA, remaps the 23 CARLA classes to 7 project
classes, and applies configurable augmentations for training.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# CARLA 23-class → project 7-class mapping
# ---------------------------------------------------------------------------
# Project classes:
#   0: Road       (CARLA 6=RoadLine, 7=Road)
#   1: Sidewalk   (CARLA 8)
#   2: Vehicle    (CARLA 10)
#   3: Pedestrian (CARLA 4)
#   4: Building   (CARLA 1, 2=Fence, 11=Wall)
#   5: Vegetation (CARLA 9, 22=Terrain)
#   6: Other      (everything else: 0,3,5,12-21)

CARLA_TO_PROJECT: Dict[int, int] = {
    0: 6,  # Unlabeled → Other
    1: 4,  # Building → Building
    2: 4,  # Fence → Building
    3: 6,  # Other → Other
    4: 3,  # Pedestrian → Pedestrian
    5: 6,  # Pole → Other
    6: 0,  # RoadLine → Road
    7: 0,  # Road → Road
    8: 1,  # Sidewalk → Sidewalk
    9: 5,  # Vegetation → Vegetation
    10: 2,  # Vehicle → Vehicle
    11: 4,  # Wall → Building
    12: 6,  # TrafficSign → Other
    13: 6,  # Sky → Other
    14: 6,  # Ground → Other
    15: 6,  # Bridge → Other
    16: 6,  # RailTrack → Other
    17: 6,  # GuardRail → Other
    18: 6,  # TrafficLight → Other
    19: 6,  # Static → Other
    20: 6,  # Dynamic → Other
    21: 6,  # Water → Other
    22: 5,  # Terrain → Vegetation
}

NUM_CLASSES = 7

# ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_class_lut() -> np.ndarray:
    """Build a 256-entry lookup table for fast class remapping.

    Returns:
        NumPy array of shape (256,) mapping CARLA class IDs to project IDs.
        Unknown IDs default to 6 (Other).
    """
    lut = np.full(256, 6, dtype=np.uint8)  # default: Other
    for carla_id, project_id in CARLA_TO_PROJECT.items():
        lut[carla_id] = project_id
    return lut


CLASS_LUT = build_class_lut()


class JointTransform:
    """Geometric + color augmentations applied consistently to image and mask.

    Geometric transforms (horizontal flip) are applied to both image and mask.
    Color transforms (brightness/contrast jitter) are applied to image only.

    Args:
        augment: Whether to apply augmentations (True for train, False for val/test).
        flip_prob: Probability of horizontal flip.
        brightness_range: (min, max) brightness jitter factor.
        contrast_range: (min, max) contrast jitter factor.
    """

    def __init__(
        self,
        augment: bool = False,
        flip_prob: float = 0.5,
        brightness_range: Tuple[float, float] = (0.8, 1.2),
        contrast_range: Tuple[float, float] = (0.8, 1.2),
    ) -> None:
        self.augment = augment
        self.flip_prob = flip_prob
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range

    def __call__(
        self, image: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        """Apply transforms to an image-mask pair.

        Args:
            image: RGB PIL Image.
            mask: Segmentation mask PIL Image (mode "L").

        Returns:
            Tuple of (transformed_image, transformed_mask).
        """
        if not self.augment:
            return image, mask

        # Geometric: horizontal flip (applied to both)
        if np.random.random() < self.flip_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Color jitter: brightness (image only)
        from PIL import ImageEnhance

        brightness_factor = np.random.uniform(*self.brightness_range)
        image = ImageEnhance.Brightness(image).enhance(brightness_factor)

        # Color jitter: contrast (image only)
        contrast_factor = np.random.uniform(*self.contrast_range)
        image = ImageEnhance.Contrast(image).enhance(contrast_factor)

        return image, mask


class CARLASegmentationDataset(Dataset):  # type: ignore[type-arg]
    """PyTorch Dataset for CARLA RGB + segmentation frame pairs.

    Discovers frame pairs across scenario subdirectories, applies a
    deterministic train/val/test split, resizes images, remaps segmentation
    classes via a NumPy LUT, and optionally applies augmentations.

    Args:
        data_dir: Root directory containing scenario subdirectories.
        split: One of "train", "val", or "test".
        image_size: (width, height) to resize images to.
        transform: Optional JointTransform for augmentation.
    """

    SPLIT_RATIOS = {"train": 0.80, "val": 0.15, "test": 0.05}

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = (512, 256),
        transform: Optional[JointTransform] = None,
    ) -> None:
        if split not in self.SPLIT_RATIOS:
            raise ValueError(f"split must be one of {list(self.SPLIT_RATIOS.keys())}")

        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size  # (width, height)
        self.transform = transform

        all_pairs = self._discover_pairs()
        self.pairs = self._apply_split(all_pairs, split)

    def _discover_pairs(self) -> List[Tuple[Path, Path]]:
        """Find all RGB + segmentation frame pairs across subdirectories.

        Returns:
            Sorted list of (rgb_path, seg_path) tuples.
        """
        pairs: List[Tuple[Path, Path]] = []
        rgb_files = sorted(self.data_dir.rglob("*_rgb.png"))
        for rgb_path in rgb_files:
            seg_path = rgb_path.parent / rgb_path.name.replace("_rgb.png", "_seg.png")
            if seg_path.exists():
                pairs.append((rgb_path, seg_path))
        return pairs

    def _apply_split(
        self, pairs: List[Tuple[Path, Path]], split: str
    ) -> List[Tuple[Path, Path]]:
        """Apply deterministic index-based split.

        Pairs are sorted by path, then partitioned by index into
        80% train / 15% val / 5% test.

        Args:
            pairs: Sorted list of frame pairs.
            split: Which split to return.

        Returns:
            Subset of pairs for the requested split.
        """
        n = len(pairs)
        train_end = int(n * self.SPLIT_RATIOS["train"])
        val_end = train_end + int(n * self.SPLIT_RATIOS["val"])

        if split == "train":
            return pairs[:train_end]
        elif split == "val":
            return pairs[train_end:val_end]
        else:  # test
            return pairs[val_end:]

    def __len__(self) -> int:
        """Return number of frame pairs in this split."""
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load and process a single frame pair.

        Args:
            idx: Index into the split.

        Returns:
            Tuple of (rgb_tensor[3,H,W] float32, mask_tensor[H,W] long).
        """
        rgb_path, seg_path = self.pairs[idx]

        # Load images
        rgb = Image.open(rgb_path).convert("RGB")
        seg = Image.open(seg_path).convert("L")

        # Resize (BILINEAR for RGB, NEAREST for mask to preserve class IDs)
        w, h = self.image_size
        rgb = rgb.resize((w, h), Image.BILINEAR)
        seg = seg.resize((w, h), Image.NEAREST)

        # Apply augmentations (if any)
        if self.transform is not None:
            rgb, seg = self.transform(rgb, seg)

        # Convert RGB to tensor and normalize with ImageNet stats
        rgb_np = np.array(rgb, dtype=np.float32) / 255.0  # (H, W, 3)
        for c in range(3):
            rgb_np[:, :, c] = (rgb_np[:, :, c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
        rgb_tensor = torch.from_numpy(rgb_np.transpose(2, 0, 1))  # (3, H, W)

        # Remap segmentation classes via LUT
        seg_np = np.array(seg, dtype=np.uint8)
        seg_remapped = CLASS_LUT[seg_np]
        mask_tensor = torch.from_numpy(seg_remapped.astype(np.int64))  # (H, W)

        return rgb_tensor, mask_tensor


def get_dataloaders(
    data_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    image_size: Tuple[int, int] = (512, 256),
) -> Dict[str, Any]:
    """Create train/val/test DataLoaders with appropriate augmentation.

    Args:
        data_dir: Root directory containing scenario subdirectories.
        batch_size: Batch size for all loaders.
        num_workers: Number of data loading workers.
        image_size: (width, height) to resize images to.

    Returns:
        Dictionary with keys "train", "val", "test" mapping to DataLoaders.
    """
    loaders: Dict[str, Any] = {}
    for split in ["train", "val", "test"]:
        augment = split == "train"
        transform = JointTransform(augment=augment)
        dataset = CARLASegmentationDataset(
            data_dir=data_dir,
            split=split,
            image_size=image_size,
            transform=transform,
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders
