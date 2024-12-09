from dataclasses import dataclass
from pathlib import Path
from typing import Optional

@dataclass
class CondensationParams:
    time_window: float = 0.2
    min_confidence: float = 0.3
    max_position_gap: float = 2.0

@dataclass
class CondensationConfig:
    """Configuration for frame condensation process"""
    time_window: float = 0.2        # Time window for grouping frames
    min_confidence: float = 0.3     # Minimum confidence threshold
    max_position_gap: float = 2.0   # Maximum allowed position gap
    output_dir: Optional[Path] = None  # Output directory for results
    
    # Additional configuration parameters
    sensor_thresholds: dict = None
    
    def __post_init__(self):
        """Initialize default values after creation"""
        if self.sensor_thresholds is None:
            self.sensor_thresholds = {
                'visibility': 0.1,
                'distance': 2.0,
                'lidar_points': 10
            }

    @classmethod
    def create_default(cls, output_dir: Path = None) -> 'CondensationConfig':
        """Create default configuration"""
        return cls(
            time_window=0.2,
            min_confidence=0.3,
            max_position_gap=2.0,
            output_dir=output_dir
        )

# Make both classes available for import
__all__ = ['CondensationConfig', 'CondensationParams']