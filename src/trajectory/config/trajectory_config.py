from dataclasses import dataclass
from typing import Optional

@dataclass
class TrajectoryConfig:
    """Configuration for trajectory analysis"""
    prediction_horizon: float = 2.0      # seconds to predict ahead
    min_frames: int = 3                  # minimum frames needed for analysis
    confidence_threshold: float = 0.7     # minimum confidence score
    max_prediction_error: float = 1.0     # meters
    time_step: float = 0.1               # seconds between predictions
    
    # Motion analysis parameters
    min_velocity: float = 0.1            # m/s, minimum to consider moving
    min_turn_angle: float = 0.1          # radians, minimum to consider turning
    min_acceleration: float = 0.5        # m/s², minimum to consider accelerating

@dataclass
class VisualizationConfig:
    """Configuration for trajectory visualization"""
    plot_width: int = 800
    plot_height: int = 600
    trajectory_color: str = '#1f77b4'
    prediction_color: str = '#ff7f0e'
    uncertainty_alpha: float = 0.3