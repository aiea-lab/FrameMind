import numpy as np
from typing import Dict
from ..config.trajectory_config import TrajectoryConfig

class MotionPatternAnalyzer:
    def __init__(self, config: TrajectoryConfig):
        self.config = config
    
    def analyze_patterns(self, motion_data: Dict) -> Dict:
        """Analyze motion patterns from trajectory data"""
        positions = motion_data['positions']
        velocities = motion_data['velocities']
        
        # Calculate speeds
        speeds = np.linalg.norm(velocities, axis=1)
        
        # Calculate direction changes
        directions = np.arctan2(velocities[:, 1], velocities[:, 0])
        direction_changes = np.diff(directions)
        
        # Determine motion type
        is_stationary = np.mean(speeds) < self.config.min_velocity
        is_turning = np.any(np.abs(direction_changes) > self.config.min_turn_angle)
        is_accelerating = np.mean(np.diff(speeds)) > self.config.min_acceleration
        
        return {
            'avg_speed': float(np.mean(speeds)),
            'max_speed': float(np.max(speeds)),
            'is_stationary': bool(is_stationary),
            'is_turning': bool(is_turning),
            'is_accelerating': bool(is_accelerating),
            'positions': positions.tolist(),
            'velocities': velocities.tolist(),
            'direction_changes': direction_changes.tolist()
        }