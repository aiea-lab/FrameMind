import numpy as np
from typing import List, Dict, Optional
from ..config.trajectory_config import TrajectoryConfig
from .motion_patterns import MotionPatternAnalyzer
from .path_predictor import PathPredictor

class TrajectoryAnalyzer:
    def __init__(self, config: TrajectoryConfig):
        self.config = config
        self.motion_analyzer = MotionPatternAnalyzer(config)
        self.path_predictor = PathPredictor(config)

    def analyze_single_object(self, obj_frame: Dict) -> Dict:
        """Analyze trajectory for a single object frame"""
        try:
            # Extract motion data
            motion_metadata = obj_frame.get('motion_metadata', [{}])[0]
            
            # Get position and velocity
            position = np.array(motion_metadata.get('location', [0, 0, 0]))
            velocity = np.array(motion_metadata.get('velocity', [0, 0, 0]))
            
            # Calculate basic metrics
            speed = np.linalg.norm(velocity)
            
            # Determine motion type
            motion_type = self._determine_motion_type(speed, velocity)
            
            # Calculate predictions
            predictions = self._predict_future_motion(position, velocity)
            
            return {
                'motion_data': {
                    'average_speed': float(speed),
                    'total_distance': float(np.linalg.norm(position)),
                    'motion_type': motion_type
                },
                'predictions': {
                    'future_positions': predictions.tolist(),
                    'confidence': float(motion_metadata.get('sensor_fusion_confidence', 0)),
                    'predicted_distance': float(np.linalg.norm(predictions[-1] - position))
                }
            }
            
        except Exception as e:
            print(f"Error analyzing object frame: {e}")
            return {}
    def _determine_motion_type(self, speed: float, velocity: np.ndarray) -> str:
        """Determine the type of motion"""
        if speed < 0.1:
            return "Stationary"
        elif np.all(np.abs(velocity[0:2]) > 0.1):
            return "Complex Motion"
        elif abs(velocity[0]) > abs(velocity[1]):
            return "Primarily Longitudinal"
        else:
            return "Primarily Lateral"

    def _predict_future_motion(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        """Predict future positions"""
        future_times = np.linspace(0, self.config.prediction_horizon, 10)
        predictions = np.array([
            position + velocity * t for t in future_times
        ])
        return predictions
    
    def analyze_trajectory(self, frames: List[Dict]) -> Dict:
        """Main trajectory analysis"""
        if len(frames) < self.config.min_frames:
            return {}
            
        # Extract motion data
        motion_data = self._extract_motion_data(frames)
        
        # Analyze motion patterns
        patterns = self.motion_analyzer.analyze_patterns(motion_data)
        
        # Generate predictions
        predictions = self.path_predictor.predict_path(motion_data)
        
        return {
            'object_id': frames[0]['object_id'],
            'object_category': frames[0].get('category', 'unknown'),
            'motion_patterns': patterns,
            'predictions': predictions
        }
    
    def _extract_motion_data(self, frames: List[Dict]) -> Dict:
        """Extract motion data from frames"""
        timestamps = []
        positions = []
        velocities = []
        
        for frame in frames:
            meta = frame['motion_metadata'][0]
            timestamps.append(float(meta['timestamp'].split('.')[-1]))
            positions.append(meta['location'])
            velocities.append(meta.get('velocity', [0, 0, 0]))
        
        return {
            'timestamps': np.array(timestamps),
            'positions': np.array(positions),
            'velocities': np.array(velocities)
        }