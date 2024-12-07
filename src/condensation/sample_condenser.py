from typing import List, Dict
from .base_condenser import BaseCondenser
import logging

logger = logging.getLogger(__name__)

class SampleCondenser(BaseCondenser):
    """Condenser for NuScenes sample frames"""
    
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        """
        Condense sample frames with sensor data.
        
        Args:
            frames: List of frame dictionaries containing raw sensor data
            
        Returns:
            List of condensed frame dictionaries
            
        Raises:
            ValueError: If frames don't contain required data
        """
        if not frames:
            return []
            
        valid_frames = []
        for i, frame in enumerate(frames):
            if 'raw_sensor_data' not in frame:
                raise ValueError(f"Frame at index {i} is missing 'raw_sensor_data': {frame}")
                
            # Convert raw_sensor_data to expected format
            processed_frame = frame.copy()
            processed_frame['sensor_data'] = {
                'lidar_path': frame['raw_sensor_data'].get('raw_lidar_path'),
                'camera_paths': frame['raw_sensor_data'].get('raw_camera_paths', {})
            }
            valid_frames.append(processed_frame)
            
        # For sample condenser, we're not actually condensing frames
        # but rather ensuring they're in the correct format
        return valid_frames

    def _can_merge_frames(self, frame1: Dict, frame2: Dict) -> bool:
        """
        Check if frames can be merged. For samples, we generally don't merge.
        """
        return False

    def _merge_frames(self, frames: List[Dict]) -> Dict:
        """
        Merge frames. For samples, we return the first frame as we don't merge.
        """
        return frames[0] if frames else None