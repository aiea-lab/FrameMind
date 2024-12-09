import numpy as np
from typing import Dict, List, Optional

class ConfidenceMetrics:
    """Calculate confidence metrics for object detection and tracking"""
    
    @staticmethod
    def calculate_point_based_confidence(lidar_points: int, radar_points: int) -> float:
        """Calculate confidence based on number of sensor points"""
        return min(1.0, (lidar_points + radar_points) / 150)
    
    @staticmethod
    def calculate_distance_confidence(distance: float, max_range: float = 100.0) -> float:
        """Calculate confidence based on distance from ego vehicle"""
        return max(0.1, 1 - (distance / max_range))
    
    @staticmethod
    def calculate_visibility_confidence(
        visibility: float,
        occlusion_status: str
    ) -> float:
        """Calculate confidence based on visibility and occlusion"""
        occlusion_weights = {
            'Visible': 1.0,
            'Partially Occluded': 0.7,
            'Mostly Occluded': 0.3
        }
        occlusion_factor = occlusion_weights.get(occlusion_status, 0.5)
        return visibility * occlusion_factor
    
    @classmethod
    def calculate_comprehensive_confidence(cls, detection_data: Dict) -> float:
        """Calculate overall confidence using multiple factors"""
        point_conf = cls.calculate_point_based_confidence(
            detection_data.get('raw_num_lidar_points', 0),
            detection_data.get('raw_num_radar_points', 0)
        )
        
        dist_conf = cls.calculate_distance_confidence(
            detection_data.get('derived_distance_to_ego', 100)
        )
        
        vis_conf = cls.calculate_visibility_confidence(
            detection_data.get('derived_visibility', 0),
            detection_data.get('derived_occlusion_status', 'Partially Occluded')
        )
        
        # Weighted combination
        weights = {'point': 0.4, 'distance': 0.3, 'visibility': 0.3}
        final_confidence = (
            weights['point'] * point_conf +
            weights['distance'] * dist_conf +
            weights['visibility'] * vis_conf
        )
        
        return final_confidence