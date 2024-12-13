from typing import List, Dict, Optional
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

class TrajectoryGenerator:
    """Generate trajectories from condensed frame data"""
    
    def __init__(self, min_points: int = 3, prediction_horizon: float = 2.0):
        self.min_points = min_points
        self.prediction_horizon = prediction_horizon

    def generate_trajectories(self, condensed_output: Dict) -> Dict:
        """Generate trajectories from condensed output"""
        trajectories = {}
        
        for frame in condensed_output.get('condensed_frames', []):
            object_id = frame['object_info']['object_id']
            
            # Extract trajectory data
            trajectory = self._process_single_trajectory(frame)
            if trajectory:
                trajectories[object_id] = trajectory
                
        return trajectories

    def _process_single_trajectory(self, frame: Dict, ego_vehicle_global_position: Optional[List[float]] = None) -> Optional[Dict]:
        """Process trajectory for a single object."""
        try:
            positions = []
            velocities = []
            timestamps = []

            # Extract tracking info
            for pos_data in frame['tracking_info']['positions']:
                positions.append(pos_data['location'])
                velocities.append(pos_data['velocity'])
                timestamps.append(pos_data['timestamp'])

            if len(positions) < self.min_points:
                return None

            positions = np.array(positions)
            velocities = np.array(velocities)

            # Apply offsets to positions if ego vehicle position is provided
            if ego_vehicle_global_position is not None:
                if len(ego_vehicle_global_position) == 3 and positions.shape[1] == 3:
                    positions -= ego_vehicle_global_position  # Apply offsets for x, y, z
                elif len(ego_vehicle_global_position) == 2 and positions.shape[1] == 3:
                    positions[:, :2] -= ego_vehicle_global_position  # Offset only x and y
                else:
                    raise ValueError("Mismatch in dimensions between positions and ego_vehicle_global_position.")

            # Apply scaling to positions (optional, for visualization purposes)
            scale_factor = 10  # Example scaling factor
            positions /= scale_factor

            # Calculate trajectory metrics
            motion_metrics = self._calculate_motion_metrics(positions, velocities)

            return {
                "object_info": {
                    "object_id": frame['object_info']['object_id'],
                    "category": frame['object_info']['category'],
                    "size": frame['object_info']['static_size']
                },
                "trajectory": {
                    "positions": positions.tolist(),
                    "velocities": velocities.tolist(),
                    "timestamps": timestamps,
                    "start_time": timestamps[0],
                    "end_time": timestamps[-1]
                },
                "motion_metrics": motion_metrics,
                "predictions": self._generate_predictions(positions, velocities)
            }
        except Exception as e:
            print(f"Error processing trajectory: {e}")
            return None

    def _calculate_motion_metrics(self, positions: np.ndarray, velocities: np.ndarray) -> Dict:
        """Calculate motion metrics for trajectory"""
        try:
            # Calculate displacement and distances
            displacements = np.diff(positions, axis=0)
            distances = np.linalg.norm(displacements, axis=1)
            total_distance = np.sum(distances)
            
            # Calculate speeds
            speeds = np.linalg.norm(velocities, axis=1)
            avg_speed = np.mean(speeds)
            max_speed = np.max(speeds)
            
            # Determine motion type
            is_stationary = avg_speed < 0.5  # threshold in m/s
            
            # Calculate direction changes
            if len(displacements) > 1:
                displacement_dirs = displacements / np.linalg.norm(displacements, axis=1)[:, np.newaxis]
                direction_changes = np.arccos(np.clip(np.sum(
                    displacement_dirs[1:] * displacement_dirs[:-1], axis=1), -1, 1))
                is_turning = np.any(direction_changes > np.pi/6)  # 30 degrees threshold
            else:
                is_turning = False
            
            return {
                "total_distance": float(total_distance),
                "average_speed": float(avg_speed),
                "max_speed": float(max_speed),
                "is_stationary": bool(is_stationary),
                "is_turning": bool(is_turning),
                "motion_type": self._classify_motion(is_stationary, is_turning, avg_speed)
            }
            
        except Exception as e:
            print(f"Error calculating motion metrics: {e}")
            return {}

    def _generate_predictions(self, positions: np.ndarray, velocities: np.ndarray) -> Dict:
        """Generate simple predictions based on current trajectory"""
        try:
            if len(positions) < 2:
                return {}

            # Use last position and velocity for linear prediction
            last_pos = positions[-1]
            last_vel = velocities[-1]
            
            # Generate prediction points
            num_points = 10
            time_steps = np.linspace(0, self.prediction_horizon, num_points)
            predictions = []
            
            for t in time_steps:
                pred_pos = last_pos + last_vel * t
                predictions.append(pred_pos.tolist())

            return {
                "predicted_positions": predictions,
                "prediction_horizon": self.prediction_horizon,
                "confidence": self._calculate_prediction_confidence(positions, velocities)
            }
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return {}

    def visualize_trajectory(self, trajectory: Dict, title: str = "Trajectory View") -> None:
        """
        Visualize a trajectory as a scatter plot.
        
        Parameters:
        - trajectory: Dict containing trajectory data with 'positions' and 'timestamps'.
        - title: Title of the plot.
        """
        try:
            positions = np.array(trajectory['trajectory']['positions'])
            timestamps = trajectory['trajectory']['timestamps']

            if positions.shape[1] != 2:
                raise ValueError("Only 2D trajectories can be visualized with this method.")

            # Create scatter plot
            plt.figure(figsize=(8, 6))
            plt.scatter(positions[:, 0], positions[:, 1], c=timestamps, cmap='viridis', label="Trajectory Points")
            plt.colorbar(label="Timestamps")
            plt.xlabel("X Position")
            plt.ylabel("Y Position")
            plt.title(title)
            plt.grid(True)
            plt.legend()
            plt.show()
        except Exception as e:
            print(f"Error visualizing trajectory: {e}")

    def _classify_motion(self, is_stationary: bool, is_turning: bool, speed: float) -> str:
        """Classify the type of motion"""
        if is_stationary:
            return "stationary"
        if is_turning:
            return "turning"
        if speed > 5.0:  # threshold in m/s
            return "fast_moving"
        return "moving"

    def _calculate_prediction_confidence(self, positions: np.ndarray, velocities: np.ndarray) -> float:
        """Calculate confidence score for predictions"""
        try:
            # More consistent motion = higher confidence
            velocity_consistency = 1.0 - np.std(np.linalg.norm(velocities, axis=1)) / (np.mean(np.linalg.norm(velocities, axis=1)) + 1e-6)
            return float(np.clip(velocity_consistency, 0.0, 1.0))
        except:
            return 0.0
        
    def process_positions(self,trajectories: Dict, ego_vehicle_position: List[float]) -> Dict:
        """
        Offset all object positions relative to the ego vehicle's position.
        """
        for obj_id, traj in trajectories.items():
            positions = np.array(traj['trajectory']['positions'])
            
            # Check dimensions and handle mismatched dimensions
            if len(ego_vehicle_position) == 2 and positions.shape[1] == 3:
                positions[:, :2] -= ego_vehicle_position  # Offset x, y only
            elif len(ego_vehicle_position) == 3 and positions.shape[1] == 3:
                positions -= ego_vehicle_position  # Offset x, y, z
            else:
                raise ValueError("Mismatch in dimensions between positions and ego_vehicle_position.")

            traj['trajectory']['positions'] = positions.tolist()
        return trajectories

# After generating condensed output
