import numpy as np
from typing import List, Dict
from ..config.trajectory_config import TrajectoryConfig

class VelocityCalculator:
    def __init__(self, config: TrajectoryConfig):
        self.config = config
    
    def calculate_velocities(self, positions: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Calculate velocities from positions"""
        dt = np.diff(timestamps)
        dp = np.diff(positions, axis=0)
        
        # Calculate velocities
        velocities = dp / dt[:, np.newaxis]
        
        # Add zero velocity for first point
        return np.vstack([velocities[0], velocities])
    
    def calculate_acceleration(self, velocities: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        """Calculate accelerations from velocities"""
        dt = np.diff(timestamps)
        dv = np.diff(velocities, axis=0)
        
        # Calculate accelerations
        accelerations = dv / dt[:, np.newaxis]
        
        # Add zero acceleration for first point
        return np.vstack([accelerations[0], accelerations])