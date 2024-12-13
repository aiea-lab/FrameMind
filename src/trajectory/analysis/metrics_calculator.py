import numpy as np
from typing import Dict
from ..models.data_models import TrajectoryData

class TrajectoryMetricsCalculator:
    """Handles all trajectory metric calculations"""
    
    @staticmethod
    def calculate_motion_metrics(trajectory: TrajectoryData) -> Dict:
        """Calculate comprehensive motion metrics for a trajectory"""
        speeds = np.linalg.norm(trajectory.velocities, axis=1)
        positions_diff = np.diff(trajectory.positions, axis=0)
        heading_angles = np.arctan2(positions_diff[:, 1], positions_diff[:, 0])
        
        metrics = {
            'average_speed': np.mean(speeds),
            'max_speed': np.max(speeds),
            'min_speed': np.min(speeds),
            'speed_variance': np.var(speeds),
            'total_distance': np.sum(np.linalg.norm(positions_diff, axis=1)),
            'straight_line_distance': np.linalg.norm(
                trajectory.positions[-1] - trajectory.positions[0]
            ),
            'average_heading': np.mean(heading_angles),
            'heading_variance': np.var(heading_angles),
            'duration': (trajectory.timestamps[-1] - trajectory.timestamps[0]).total_seconds()
        }
        
        metrics['motion_type'] = TrajectoryMetricsCalculator._classify_motion_type(
            metrics['straight_line_distance'],
            metrics['total_distance'],
            metrics['heading_variance']
        )
        
        return metrics
    
    @staticmethod
    def _classify_motion_type(straight_dist: float, total_dist: float, 
                            heading_var: float) -> str:
        """Classify the type of motion based on trajectory characteristics"""
        straightness_ratio = straight_dist / total_dist if total_dist > 0 else 1
        
        if straightness_ratio > 0.95:
            return "Linear"
        elif heading_var < 0.5:
            return "Curved"
        elif heading_var < 1.5:
            return "Turning"
        else:
            return "Complex"