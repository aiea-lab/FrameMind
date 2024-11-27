from typing import List, Dict
import numpy as np
from .base_condenser import BaseCondenser, CondenserConfig

class SampleCondenser(BaseCondenser):
    def __init__(self, config: CondenserConfig):
        super().__init__(config)
    
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        if not frames:
            return []
        
        # Filter high-confidence samples
        valid_frames = self._filter_valid_samples(frames)
        
        condensed_frames = []
        current_group = []
        
        for frame in valid_frames:
            if not current_group:
                current_group.append(frame)
                continue
            
            if self._can_merge_samples(current_group[-1], frame):
                current_group.append(frame)
            else:
                merged = self._merge_sample_frames(current_group)
                condensed_frames.append(merged)
                current_group = [frame]
        
        if current_group:
            merged = self._merge_sample_frames(current_group)
            condensed_frames.append(merged)
        
        return condensed_frames
    
    def _filter_valid_samples(self, frames: List[Dict]) -> List[Dict]:
        """Filter samples based on confidence and point count"""
        return [
            frame for frame in frames
            if frame['motion_metadata'][0].get('sensor_fusion_confidence', 0) 
               >= self.config.confidence_threshold
            and frame['motion_metadata'][0].get('num_lidar_points', 0) 
               >= self.config.min_points_threshold
        ]
    
    def _can_merge_samples(self, frame1: Dict, frame2: Dict) -> bool:
        """Check if two sample frames can be merged"""
        time_diff = self.calculate_time_difference(
            frame1['motion_metadata'][0]['timestamp'],
            frame2['motion_metadata'][0]['timestamp']
        )
        return time_diff <= self.config.time_window
    
    def _merge_sample_frames(self, frames: List[Dict]) -> Dict:
        """Merge sample frames with sensor data fusion"""
        if not frames:
            return None
        
        base_frame = max(frames, 
                        key=lambda x: x['motion_metadata'][0]['sensor_fusion_confidence'])
        merged = base_frame.copy()
        metadata = merged['motion_metadata'][0]
        
        # Merge sensor data
        all_metadata = [f['motion_metadata'][0] for f in frames]
        metadata['num_lidar_points'] = sum(m.get('num_lidar_points', 0) 
                                         for m in all_metadata)
        metadata['num_radar_points'] = sum(m.get('num_radar_points', 0) 
                                         for m in all_metadata)
        
        # Update camera data if available
        if 'camera_data' in metadata:
            metadata['camera_data'] = self._merge_camera_data(
                [m.get('camera_data', {}) for m in all_metadata]
            )
        
        return merged
    
    def _merge_camera_data(self, camera_data_list: List[Dict]) -> Dict:
        """Merge camera data from multiple frames"""
        merged_data = {}
        for cam_data in camera_data_list:
            for camera, bbox in cam_data.get('bounding_boxes', {}).items():
                if camera not in merged_data:
                    merged_data[camera] = bbox
        return {'bounding_boxes': merged_data}