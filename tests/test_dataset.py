"""Unit tests for CARLASegmentationDataset and related utilities.

All tests use synthetic PNG images via tmp_path fixture — no CARLA dependency.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from src.data_collection.dataset import (
    CARLA_TO_PROJECT,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_CLASSES,
    CARLASegmentationDataset,
    JointTransform,
    build_class_lut,
)


def _create_synthetic_frames(
    base_dir, num_frames: int = 20, width: int = 64, height: int = 32
):
    """Create synthetic RGB + segmentation frame pairs for testing.

    Args:
        base_dir: Root directory to create scenario subdirectory in.
        num_frames: Number of frame pairs to create.
        width: Image width.
        height: Image height.

    Returns:
        Path to the base directory.
    """
    scenario_dir = base_dir / "Town01_ClearNoon_20260101_120000"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_frames):
        # RGB: random colors
        rgb_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        rgb_img = Image.fromarray(rgb_array, mode="RGB")
        rgb_img.save(scenario_dir / f"{i:06d}_rgb.png")

        # Segmentation: random CARLA class IDs (0-22)
        seg_array = np.random.randint(0, 23, (height, width), dtype=np.uint8)
        seg_img = Image.fromarray(seg_array, mode="L")
        seg_img.save(scenario_dir / f"{i:06d}_seg.png")

    return base_dir


class TestClassMapping:
    """Tests for CARLA → project class mapping."""

    def test_all_23_carla_ids_covered(self):
        """All 23 CARLA class IDs (0-22) are present in the mapping."""
        for carla_id in range(23):
            assert carla_id in CARLA_TO_PROJECT, f"CARLA ID {carla_id} not in mapping"

    def test_outputs_in_valid_range(self):
        """All mapped values are in [0, 6]."""
        for carla_id, project_id in CARLA_TO_PROJECT.items():
            assert 0 <= project_id < NUM_CLASSES, (
                f"CARLA ID {carla_id} maps to {project_id}, "
                f"outside [0, {NUM_CLASSES - 1}]"
            )

    def test_lut_matches_dict(self):
        """LUT produces the same results as the dictionary mapping."""
        lut = build_class_lut()
        for carla_id, project_id in CARLA_TO_PROJECT.items():
            assert (
                lut[carla_id] == project_id
            ), f"LUT[{carla_id}]={lut[carla_id]} != dict[{carla_id}]={project_id}"

    def test_unknown_ids_map_to_other(self):
        """IDs outside 0-22 map to Other (6)."""
        lut = build_class_lut()
        for unknown_id in [23, 50, 100, 200, 255]:
            assert lut[unknown_id] == 6, (
                f"Unknown ID {unknown_id} should map to 6 (Other), "
                f"got {lut[unknown_id]}"
            )

    def test_specific_mappings(self):
        """Verify key class mappings are correct."""
        assert CARLA_TO_PROJECT[7] == 0  # Road → Road
        assert CARLA_TO_PROJECT[6] == 0  # RoadLine → Road
        assert CARLA_TO_PROJECT[8] == 1  # Sidewalk → Sidewalk
        assert CARLA_TO_PROJECT[10] == 2  # Vehicle → Vehicle
        assert CARLA_TO_PROJECT[4] == 3  # Pedestrian → Pedestrian
        assert CARLA_TO_PROJECT[1] == 4  # Building → Building
        assert CARLA_TO_PROJECT[9] == 5  # Vegetation → Vegetation
        assert CARLA_TO_PROJECT[22] == 5  # Terrain → Vegetation

    def test_lut_shape_and_dtype(self):
        """LUT has correct shape and dtype."""
        lut = build_class_lut()
        assert lut.shape == (256,)
        assert lut.dtype == np.uint8


class TestDatasetSplit:
    """Tests for dataset train/val/test splitting."""

    def test_splits_are_disjoint(self, tmp_path):
        """Train, val, and test splits have no overlapping samples."""
        _create_synthetic_frames(tmp_path, num_frames=100)

        train_ds = CARLASegmentationDataset(str(tmp_path), split="train")
        val_ds = CARLASegmentationDataset(str(tmp_path), split="val")
        test_ds = CARLASegmentationDataset(str(tmp_path), split="test")

        train_paths = {str(p[0]) for p in train_ds.pairs}
        val_paths = {str(p[0]) for p in val_ds.pairs}
        test_paths = {str(p[0]) for p in test_ds.pairs}

        assert train_paths.isdisjoint(val_paths), "Train and val overlap"
        assert train_paths.isdisjoint(test_paths), "Train and test overlap"
        assert val_paths.isdisjoint(test_paths), "Val and test overlap"

    def test_full_coverage(self, tmp_path):
        """All samples are assigned to exactly one split."""
        _create_synthetic_frames(tmp_path, num_frames=100)

        train_ds = CARLASegmentationDataset(str(tmp_path), split="train")
        val_ds = CARLASegmentationDataset(str(tmp_path), split="val")
        test_ds = CARLASegmentationDataset(str(tmp_path), split="test")

        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == 100, f"Expected 100, got {total}"

    def test_approximate_ratios(self, tmp_path):
        """Split ratios approximately match 80/15/5."""
        _create_synthetic_frames(tmp_path, num_frames=100)

        train_ds = CARLASegmentationDataset(str(tmp_path), split="train")
        val_ds = CARLASegmentationDataset(str(tmp_path), split="val")
        test_ds = CARLASegmentationDataset(str(tmp_path), split="test")

        assert len(train_ds) == 80
        assert len(val_ds) == 15
        assert len(test_ds) == 5

    def test_splits_are_deterministic(self, tmp_path):
        """Creating splits twice yields the same assignment."""
        _create_synthetic_frames(tmp_path, num_frames=50)

        ds1 = CARLASegmentationDataset(str(tmp_path), split="train")
        ds2 = CARLASegmentationDataset(str(tmp_path), split="train")

        paths1 = [str(p[0]) for p in ds1.pairs]
        paths2 = [str(p[0]) for p in ds2.pairs]
        assert paths1 == paths2

    def test_invalid_split_raises(self, tmp_path):
        """Invalid split name raises ValueError."""
        _create_synthetic_frames(tmp_path, num_frames=10)
        with pytest.raises(ValueError, match="split must be one of"):
            CARLASegmentationDataset(str(tmp_path), split="invalid")


class TestDatasetLoading:
    """Tests for dataset __getitem__ output shapes and dtypes."""

    def test_output_shapes(self, tmp_path):
        """RGB tensor is (3,256,512) and mask tensor is (256,512)."""
        _create_synthetic_frames(tmp_path, num_frames=5)
        ds = CARLASegmentationDataset(str(tmp_path), split="train")

        if len(ds) == 0:
            pytest.skip("No training samples (too few frames)")

        rgb, mask = ds[0]
        assert rgb.shape == (3, 256, 512), f"RGB shape: {rgb.shape}"
        assert mask.shape == (256, 512), f"Mask shape: {mask.shape}"

    def test_output_dtypes(self, tmp_path):
        """RGB is float32, mask is long (int64)."""
        _create_synthetic_frames(tmp_path, num_frames=5)
        ds = CARLASegmentationDataset(str(tmp_path), split="train")

        if len(ds) == 0:
            pytest.skip("No training samples (too few frames)")

        rgb, mask = ds[0]
        assert rgb.dtype == torch.float32
        assert mask.dtype == torch.int64

    def test_mask_values_in_range(self, tmp_path):
        """All mask values are in [0, 6]."""
        _create_synthetic_frames(tmp_path, num_frames=5)
        ds = CARLASegmentationDataset(str(tmp_path), split="train")

        if len(ds) == 0:
            pytest.skip("No training samples (too few frames)")

        _, mask = ds[0]
        assert mask.min() >= 0, f"Mask min: {mask.min()}"
        assert mask.max() < NUM_CLASSES, f"Mask max: {mask.max()}"

    def test_imagenet_normalization_applied(self, tmp_path):
        """RGB values are approximately in ImageNet-normalized range."""
        _create_synthetic_frames(tmp_path, num_frames=5)
        ds = CARLASegmentationDataset(str(tmp_path), split="train")

        if len(ds) == 0:
            pytest.skip("No training samples (too few frames)")

        rgb, _ = ds[0]

        # After ImageNet normalization, values should be roughly in [-2.5, 2.5]
        # (not [0, 255] or [0, 1])
        assert rgb.min() < 0, "Expected negative values after normalization"
        assert rgb.max() < 10, "Values too large — normalization may not be applied"

        # Check that the mean of each channel is approximately
        # (0.5 - mean) / std for uniform [0,1] input
        for c in range(3):
            channel_mean = rgb[c].mean().item()
            expected_approx = (0.5 - IMAGENET_MEAN[c]) / IMAGENET_STD[c]
            assert abs(channel_mean - expected_approx) < 1.0, (
                f"Channel {c} mean {channel_mean:.2f} far from "
                f"expected ~{expected_approx:.2f}"
            )


class TestTransforms:
    """Tests for JointTransform augmentation behavior."""

    def test_flip_applied_to_both(self, tmp_path):
        """Horizontal flip is applied to both image and mask consistently."""
        # Create asymmetric images to detect flips
        width, height = 64, 32

        # RGB with gradient (left dark, right bright)
        rgb_array = np.zeros((height, width, 3), dtype=np.uint8)
        rgb_array[:, width // 2 :, :] = 255
        rgb_img = Image.fromarray(rgb_array, mode="RGB")

        # Mask with gradient
        mask_array = np.zeros((height, width), dtype=np.uint8)
        mask_array[:, width // 2 :] = 7
        mask_img = Image.fromarray(mask_array, mode="L")

        # Force flip (probability 1.0)
        transform = JointTransform(augment=True, flip_prob=1.0)
        flipped_rgb, flipped_mask = transform(rgb_img, mask_img)

        flipped_rgb_np = np.array(flipped_rgb)
        flipped_mask_np = np.array(flipped_mask)

        # After flip, left side should be bright, right side dark
        assert flipped_rgb_np[:, 0, 0].mean() > 200, "Left should be bright after flip"
        assert flipped_rgb_np[:, -1, 0].mean() < 50, "Right should be dark after flip"

        # Mask should also be flipped
        assert flipped_mask_np[:, 0].mean() > 5, "Left mask should have high values"
        assert flipped_mask_np[:, -1].mean() < 2, "Right mask should have low values"

    def test_no_augmentation_on_val(self):
        """Val transform does not modify images."""
        width, height = 64, 32

        rgb_array = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        rgb_img = Image.fromarray(rgb_array, mode="RGB")

        mask_array = np.random.randint(0, 7, (height, width), dtype=np.uint8)
        mask_img = Image.fromarray(mask_array, mode="L")

        transform = JointTransform(augment=False)
        out_rgb, out_mask = transform(rgb_img, mask_img)

        np.testing.assert_array_equal(np.array(out_rgb), rgb_array)
        np.testing.assert_array_equal(np.array(out_mask), mask_array)

    def test_color_jitter_does_not_affect_mask(self):
        """Color jitter changes only the RGB image, not the mask."""
        width, height = 64, 32

        rgb_array = np.full((height, width, 3), 128, dtype=np.uint8)
        rgb_img = Image.fromarray(rgb_array, mode="RGB")

        mask_array = np.full((height, width), 5, dtype=np.uint8)
        mask_img = Image.fromarray(mask_array, mode="L")

        # Force no flip, but extreme color jitter
        transform = JointTransform(
            augment=True,
            flip_prob=0.0,
            brightness_range=(1.5, 1.5),
            contrast_range=(1.5, 1.5),
        )
        _, out_mask = transform(rgb_img, mask_img)

        # Mask should remain unchanged
        np.testing.assert_array_equal(np.array(out_mask), mask_array)
