from dataclasses import dataclass
from pathlib import Path

@dataclass
class CondensationConfig:
    """Configuration for frame condensation"""
    # Time parameters
    time_window: float = 0.1          # seconds
    min_time_gap: float = 0.02        # minimum time between frames
    max_time_gap: float = 0.5         # maximum time gap to consider

    # Quality thresholds
    min_confidence: float = 0.7       # minimum confidence score
    min_visibility: float = 0.3       # minimum visibility score
    min_lidar_points: int = 5         # minimum LiDAR points

    # Motion parameters
    max_position_gap: float = 0.5     # meters
    max_velocity: float = 40.0        # m/s
    max_acceleration: float = 10.0    # m/s²

    # Scene parameters
    max_scene_objects: int = 50       # maximum objects in scene
    min_scene_coverage: float = 0.6   # minimum scene coverage

    # Output settings
    output_dir: Path = Path("output/condensed")