import numpy as np
from typing import Dict
from ..config.trajectory_config import TrajectoryConfig

class PathPredictor:
    def __init__(self, config: TrajectoryConfig):
        self.config = config
    
    def predict_path(self, motion_data: Dict) -> Dict:
        """Predict future path based on current motion"""
        positions = motion_data['positions']
        velocities = motion_data['velocities']
        timestamps = motion_data['timestamps']
        
        # Use last position and velocity
        last_pos = positions[-1]
        last_vel = velocities[-1]
        last_time = timestamps[-1]
        
        # Generate future timestamps
        future_times = np.linspace(
            last_time,
            last_time + self.config.prediction_horizon,
            20
        )
        time_deltas = future_times - last_time
        
        # Predict future positions
        future_positions = np.array([
            last_pos + last_vel * dt for dt in time_deltas
        ])
        
        return {
            'future_positions': future_positions.tolist(),
            'prediction_horizon': self.config.prediction_horizon,
            'confidence': self._calculate_prediction_confidence(motion_data)
        }
    
    def _calculate_prediction_confidence(self, motion_data: Dict) -> float:
        """Calculate confidence in prediction"""
        # Simple confidence based on motion consistency
        velocities = motion_data['velocities']
        velocity_consistency = 1.0 - np.std(np.linalg.norm(velocities, axis=1)) / \
                             (np.mean(np.linalg.norm(velocities, axis=1)) + 1e-6)
        
        return float(max(0.0, min(1.0, velocity_consistency)))