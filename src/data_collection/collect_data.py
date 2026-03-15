"""CARLA data collection script with synchronized RGB + segmentation capture.

Connects to a running CARLA server, spawns an ego vehicle with RGB and semantic
segmentation cameras, and saves synchronized frame pairs to disk. Supports
multiple scenarios via config or CLI arguments.
"""

import argparse
import json
import queue
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml
from PIL import Image

try:
    import carla
except ImportError:
    carla = None  # type: ignore[assignment]


class DataCollector:
    """Collects synchronized RGB + segmentation frames from CARLA.

    Manages the lifecycle of CARLA actors (vehicle, cameras, NPCs) and
    captures frame pairs using queue-based synchronization.

    Args:
        host: CARLA server hostname.
        port: CARLA server port.
        timeout: Connection timeout in seconds.
        output_dir: Root directory for saving collected data.
        camera_position: Camera mount position relative to vehicle.
        warmup_ticks: Number of simulation ticks to skip before recording.
        save_metadata: Whether to save metadata JSON alongside frames.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 2000,
        timeout: float = 10.0,
        output_dir: str = "data/raw",
        camera_position: Optional[Dict[str, float]] = None,
        warmup_ticks: int = 30,
        save_metadata: bool = True,
    ) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.output_dir = Path(output_dir)
        self.camera_position = camera_position or {"x": 1.5, "y": 0.0, "z": 2.4}
        self.warmup_ticks = warmup_ticks
        self.save_metadata = save_metadata

        self.client: Any = None
        self.world: Any = None
        self.ego_vehicle: Any = None
        self.rgb_camera: Any = None
        self.seg_camera: Any = None
        self.actors: List[Any] = []
        self._original_settings: Any = None

    def connect(self) -> None:
        """Connect to the CARLA server.

        Raises:
            RuntimeError: If carla package is not installed.
            Exception: If connection to CARLA server fails.
        """
        if carla is None:
            raise RuntimeError(
                "carla package not installed. "
                "Install it from the CARLA PythonAPI egg/wheel."
            )
        self.client = carla.Client(self.host, self.port)
        self.client.set_timeout(self.timeout)
        print(f"Connected to CARLA server at {self.host}:{self.port}")

    def setup_world(self, map_name: str, weather_preset: str) -> None:
        """Load a map and configure world settings.

        Args:
            map_name: CARLA map name (e.g., "Town01").
            weather_preset: Weather preset name (e.g., "ClearNoon").
        """
        self.client.load_world(map_name)
        self.world = self.client.get_world()

        # Save original settings for cleanup
        self._original_settings = self.world.get_settings()

        # Enable synchronous mode
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        self.world.apply_settings(settings)

        # Apply weather
        weather = getattr(carla.WeatherParameters, weather_preset)
        self.world.set_weather(weather)
        print(f"World setup: map={map_name}, weather={weather_preset}, sync=True")

    def spawn_ego_vehicle(self) -> None:
        """Spawn the ego vehicle (Tesla Model 3) with autopilot enabled."""
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.find("vehicle.tesla.model3")

        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No spawn points available on this map")

        self.ego_vehicle = self.world.spawn_actor(vehicle_bp, spawn_points[0])
        self.ego_vehicle.set_autopilot(True)
        self.actors.append(self.ego_vehicle)
        print("Spawned ego vehicle: Tesla Model 3 with autopilot")

    def spawn_npcs(self, num_vehicles: int = 20, num_pedestrians: int = 10) -> None:
        """Spawn NPC vehicles and pedestrians using batch API.

        Args:
            num_vehicles: Number of NPC vehicles to spawn.
            num_pedestrians: Number of pedestrians to spawn.
        """
        blueprint_library = self.world.get_blueprint_library()
        spawn_points = self.world.get_map().get_spawn_points()

        # Set traffic manager to synchronous mode
        traffic_manager = self.client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        # Spawn vehicles via batch
        vehicle_bps = blueprint_library.filter("vehicle.*")
        vehicle_spawn_points = spawn_points[1 : num_vehicles + 1]

        batch_cmds = []
        for i, sp in enumerate(vehicle_spawn_points):
            bp = vehicle_bps[i % len(vehicle_bps)]
            if bp.has_attribute("color"):
                color = np.random.choice(bp.get_attribute("color").recommended_values)
                bp.set_attribute("color", color)
            batch_cmds.append(
                carla.command.SpawnActor(bp, sp).then(
                    carla.command.SetAutopilot(carla.command.FutureActor, True, 8000)
                )
            )

        vehicle_results = self.client.apply_batch_sync(batch_cmds, True)
        vehicles_spawned = 0
        for result in vehicle_results:
            if not result.error:
                self.actors.append(self.world.get_actor(result.actor_id))
                vehicles_spawned += 1

        # Spawn pedestrians
        walker_bps = blueprint_library.filter("walker.pedestrian.*")
        peds_spawned = 0

        for _ in range(num_pedestrians):
            walker_bp = np.random.choice(walker_bps)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            spawn_point = carla.Transform()
            loc = self.world.get_random_location_from_navigation()
            if loc is None:
                continue
            spawn_point.location = loc

            walker = self.world.try_spawn_actor(walker_bp, spawn_point)
            if walker is None:
                continue

            self.actors.append(walker)

            # Spawn AI controller for pedestrian
            controller_bp = blueprint_library.find("controller.ai.walker")
            controller = self.world.spawn_actor(
                controller_bp, carla.Transform(), walker
            )
            self.actors.append(controller)

            # Tick the world so the controller can initialize
            self.world.tick()
            controller.start()
            controller.go_to_location(self.world.get_random_location_from_navigation())
            controller.set_max_speed(1.0 + np.random.random() * 1.5)
            peds_spawned += 1

        print(f"Spawned NPCs: {vehicles_spawned} vehicles, {peds_spawned} pedestrians")

    def attach_cameras(
        self, width: int = 640, height: int = 480, fov: float = 90.0
    ) -> None:
        """Attach RGB and semantic segmentation cameras to the ego vehicle.

        Both cameras share identical parameters for frame alignment.

        Args:
            width: Image width in pixels.
            height: Image height in pixels.
            fov: Field of view in degrees.
        """
        blueprint_library = self.world.get_blueprint_library()

        cam_transform = carla.Transform(
            carla.Location(
                x=self.camera_position["x"],
                y=self.camera_position["y"],
                z=self.camera_position["z"],
            )
        )

        # RGB camera
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(width))
        rgb_bp.set_attribute("image_size_y", str(height))
        rgb_bp.set_attribute("fov", str(fov))
        rgb_bp.set_attribute("sensor_tick", "0.05")
        self.rgb_camera = self.world.spawn_actor(
            rgb_bp, cam_transform, attach_to=self.ego_vehicle
        )
        self.actors.append(self.rgb_camera)

        # Semantic segmentation camera
        seg_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        seg_bp.set_attribute("image_size_x", str(width))
        seg_bp.set_attribute("image_size_y", str(height))
        seg_bp.set_attribute("fov", str(fov))
        seg_bp.set_attribute("sensor_tick", "0.05")
        self.seg_camera = self.world.spawn_actor(
            seg_bp, cam_transform, attach_to=self.ego_vehicle
        )
        self.actors.append(self.seg_camera)

        print(f"Attached cameras: {width}x{height}, FOV={fov}")

    def collect_frames(
        self,
        num_frames: int,
        scenario_dir: Path,
        width: int = 640,
        height: int = 480,
    ) -> int:
        """Collect synchronized RGB + segmentation frame pairs.

        Uses queue-based synchronization with frame ID assertion to ensure
        RGB and segmentation frames are perfectly aligned.

        Args:
            num_frames: Number of frame pairs to collect.
            scenario_dir: Directory to save frames in.
            width: Camera image width.
            height: Camera image height.

        Returns:
            Number of frames successfully saved.
        """
        scenario_dir.mkdir(parents=True, exist_ok=True)

        rgb_queue: queue.Queue[Any] = queue.Queue()
        seg_queue: queue.Queue[Any] = queue.Queue()

        self.rgb_camera.listen(rgb_queue.put)
        self.seg_camera.listen(seg_queue.put)

        # Warmup: let simulation settle
        print(f"Warming up for {self.warmup_ticks} ticks...")
        for _ in range(self.warmup_ticks):
            self.world.tick()
            try:
                rgb_queue.get(timeout=2.0)
                seg_queue.get(timeout=2.0)
            except queue.Empty:
                pass

        # Collect frames
        saved = 0
        for i in range(num_frames):
            self.world.tick()

            try:
                rgb_data = rgb_queue.get(timeout=2.0)
                seg_data = seg_queue.get(timeout=2.0)
            except queue.Empty:
                print(f"Warning: timeout waiting for frame {i}, skipping")
                continue

            # Assert frame alignment
            if rgb_data.frame != seg_data.frame:
                print(
                    f"Warning: frame mismatch rgb={rgb_data.frame} "
                    f"seg={seg_data.frame}, skipping"
                )
                continue

            # Save RGB: BGRA → RGB PNG
            rgb_array = np.frombuffer(rgb_data.raw_data, dtype=np.uint8)
            rgb_array = rgb_array.reshape((height, width, 4))
            rgb_array = rgb_array[:, :, :3][:, :, ::-1]  # BGRA → BGR → RGB
            rgb_image = Image.fromarray(rgb_array, mode="RGB")
            rgb_image.save(scenario_dir / f"{saved:06d}_rgb.png")

            # Save segmentation: BGRA → channel R (index 2) → grayscale PNG
            seg_array = np.frombuffer(seg_data.raw_data, dtype=np.uint8)
            seg_array = seg_array.reshape((height, width, 4))
            seg_class_ids = seg_array[:, :, 2]  # R channel in BGRA
            seg_image = Image.fromarray(seg_class_ids, mode="L")
            seg_image.save(scenario_dir / f"{saved:06d}_seg.png")

            saved += 1

            if (saved) % 100 == 0:
                print(f"  Saved {saved}/{num_frames} frames")

        # Stop listening
        self.rgb_camera.stop()
        self.seg_camera.stop()

        print(f"Collected {saved} frame pairs in {scenario_dir}")
        return saved

    def save_scenario_metadata(
        self,
        scenario_dir: Path,
        map_name: str,
        weather_preset: str,
        num_vehicles: int,
        num_pedestrians: int,
        num_frames: int,
        width: int = 640,
        height: int = 480,
        fov: float = 90.0,
    ) -> None:
        """Save metadata JSON for a collection scenario.

        Args:
            scenario_dir: Directory to save metadata in.
            map_name: CARLA map name.
            weather_preset: Weather preset name.
            num_vehicles: Number of NPC vehicles.
            num_pedestrians: Number of pedestrians.
            num_frames: Number of frames collected.
            width: Camera image width.
            height: Camera image height.
            fov: Camera FOV.
        """
        if not self.save_metadata:
            return

        metadata: Dict[str, Any] = {
            "map": map_name,
            "weather": weather_preset,
            "num_vehicles": num_vehicles,
            "num_pedestrians": num_pedestrians,
            "num_frames": num_frames,
            "camera": {
                "width": width,
                "height": height,
                "fov": fov,
                "position": self.camera_position,
                "sensor_tick": 0.05,
            },
            "carla_version": self.client.get_server_version(),
            "timestamp": datetime.now().isoformat(),
        }

        metadata_path = scenario_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")

    def cleanup(self) -> None:
        """Destroy all spawned actors and restore world settings."""
        print("Cleaning up actors...")
        for actor in reversed(self.actors):
            try:
                actor.destroy()
            except Exception:
                pass
        self.actors.clear()

        # Restore original settings (async mode)
        if self.world is not None and self._original_settings is not None:
            try:
                self.world.apply_settings(self._original_settings)
            except Exception:
                pass

        print("Cleanup complete")

    def run_scenario(
        self,
        map_name: str,
        weather_preset: str,
        num_frames: int,
        num_vehicles: int = 20,
        num_pedestrians: int = 10,
    ) -> None:
        """Run a single data collection scenario end-to-end.

        Args:
            map_name: CARLA map name.
            weather_preset: Weather preset name.
            num_frames: Number of frames to collect.
            num_vehicles: Number of NPC vehicles.
            num_pedestrians: Number of pedestrians.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        scenario_dir = self.output_dir / f"{map_name}_{weather_preset}_{timestamp}"

        try:
            self.setup_world(map_name, weather_preset)
            self.spawn_ego_vehicle()
            self.spawn_npcs(num_vehicles, num_pedestrians)
            self.attach_cameras()
            saved = self.collect_frames(num_frames, scenario_dir)
            self.save_scenario_metadata(
                scenario_dir,
                map_name,
                weather_preset,
                num_vehicles,
                num_pedestrians,
                saved,
            )
        finally:
            self.cleanup()


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """CLI entry point for data collection."""
    parser = argparse.ArgumentParser(
        description="Collect synchronized RGB + segmentation data from CARLA"
    )
    parser.add_argument("--map", type=str, default="Town01", help="CARLA map name")
    parser.add_argument(
        "--weather", type=str, default="ClearNoon", help="Weather preset"
    )
    parser.add_argument(
        "--frames", type=int, default=1000, help="Number of frames to collect"
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/raw", help="Output directory"
    )
    parser.add_argument(
        "--num-vehicles", type=int, default=20, help="Number of NPC vehicles"
    )
    parser.add_argument(
        "--num-pedestrians", type=int, default=10, help="Number of pedestrians"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/carla_config.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--scenario-all",
        action="store_true",
        help="Run all scenarios from config file",
    )

    args = parser.parse_args()

    # Load config for defaults
    config = load_config(args.config)
    carla_config = config.get("carla", {})
    dc_config = config.get("data_collection", {})

    host = carla_config.get("host", "localhost")
    port = carla_config.get("port", 2000)
    timeout = carla_config.get("timeout", 10.0)
    output_dir = args.output_dir or dc_config.get("output_dir", "data/raw")
    camera_position = dc_config.get("camera_position")
    warmup_ticks = dc_config.get("warmup_ticks", 30)
    save_metadata_flag = dc_config.get("save_metadata", True)

    collector = DataCollector(
        host=host,
        port=port,
        timeout=timeout,
        output_dir=output_dir,
        camera_position=camera_position,
        warmup_ticks=warmup_ticks,
        save_metadata=save_metadata_flag,
    )

    collector.connect()

    if args.scenario_all:
        scenarios = dc_config.get("scenarios", [])
        if not scenarios:
            print("No scenarios found in config file")
            sys.exit(1)

        print(f"Running {len(scenarios)} scenarios...")
        for i, scenario in enumerate(scenarios, 1):
            print(f"\n{'='*60}")
            map_name = scenario["map"]
            weather = scenario["weather"]
            print(f"Scenario {i}/{len(scenarios)}: {map_name} - {weather}")
            print(f"{'='*60}")
            collector.run_scenario(
                map_name=scenario["map"],
                weather_preset=scenario["weather"],
                num_frames=scenario.get("frames", 1000),
                num_vehicles=scenario.get("num_vehicles", 20),
                num_pedestrians=scenario.get("num_pedestrians", 10),
            )
            # Brief pause between scenarios
            time.sleep(2)
    else:
        collector.run_scenario(
            map_name=args.map,
            weather_preset=args.weather,
            num_frames=args.frames,
            num_vehicles=args.num_vehicles,
            num_pedestrians=args.num_pedestrians,
        )

    print("\nData collection complete!")


if __name__ == "__main__":
    main()
