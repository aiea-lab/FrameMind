from typing import Dict, List
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

class BinaryTrajectoryAnomalyDetector:
    def __init__(self, contamination: float = 0.1):
        """
        Initialize the binary anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset (between 0 and 0.5)
        """
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42
        )
        
    def _extract_features(self, trajectory_data: Dict) -> np.ndarray:
        """Extract relevant features from trajectory data."""
        # Extract positions from trajectory data
        positions = np.array(trajectory_data.get('trajectory', {}).get('positions', []))
        
        if len(positions) < 2:
            return np.array([])
            
        # Get motion metrics
        motion_metrics = trajectory_data.get('motion_metrics', {})
        
        features = []
        
        # Add position features
        pos_mean = positions.mean(axis=0)
        pos_std = positions.std(axis=0)
        
        # Add motion metric features
        features.extend([
            motion_metrics.get('average_speed', 0),
            motion_metrics.get('total_distance', 0),
            motion_metrics.get('average_acceleration', 0),
            1 if motion_metrics.get('motion_type') == 'stationary' else 0,
            1 if motion_metrics.get('motion_type') == 'moving' else 0
        ])
        
        # Combine all features
        features = np.concatenate([
            pos_mean, pos_std,
            np.array(features)
        ])
        
        return features.reshape(1, -1)
        
    def is_anomalous(self, trajectory_data: Dict) -> int:
        """
        Determine if a trajectory is anomalous.
        
        Args:
            trajectory_data: Dictionary containing trajectory information
            
        Returns:
            1 if trajectory is anomalous, 0 if normal
        """
        features = self._extract_features(trajectory_data)
        
        if len(features) == 0:
            return 1  # Consider too-short trajectories as anomalous
            
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Predict using Isolation Forest (-1 for anomalies, 1 for normal)
        prediction = self.isolation_forest.fit_predict(scaled_features)[0]
        
        # Convert to binary (0 for normal, 1 for anomalous)
        return 1 if prediction == -1 else 0

def detect_trajectory_anomalies(trajectory_results: Dict, contamination: float = 0.1) -> Dict:
    """
    Process all trajectories and detect anomalies.
    
    Args:
        trajectory_results: Dictionary containing trajectory data
        contamination: Expected proportion of anomalies
        
    Returns:
        Dictionary with binary anomaly labels for each object
    """
    detector = BinaryTrajectoryAnomalyDetector(contamination=contamination)
    
    results = {
        'object_anomalies': {},
        'statistics': {
            'total_objects': 0,
            'anomalous_objects': 0
        }
    }
    
    # Process each object's trajectory
    for obj_id, obj_data in trajectory_results.items():
        is_anomalous = detector.is_anomalous(obj_data)
        
        results['object_anomalies'][obj_id] = {
            'is_anomalous': is_anomalous,
            'object_category': obj_data.get('object_info', {}).get('category', 'unknown')
        }
        
        results['statistics']['total_objects'] += 1
        results['statistics']['anomalous_objects'] += is_anomalous
    
    # Add percentage
    total = results['statistics']['total_objects']
    if total > 0:
        results['statistics']['anomaly_percentage'] = (
            results['statistics']['anomalous_objects'] / total * 100
        )
    
    return results