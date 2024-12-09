from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime

class ObjectFrameCondenser:
    """Class for condensing object frames using various methods."""
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize the frame condenser with thresholds.
        
        Args:
            thresholds: Dictionary containing threshold values for different metrics
        """
        self.thresholds = thresholds or {
            'visibility': 0.1,
            'distance': 2.0,
            'lidar_points': 10
        }

    def condense_frames(self, frames: List[Dict], method: str = 'keyframe', **kwargs) -> List[Dict]:
        """
        Main method to condense frames using specified method.
        
        Args:
            frames: List of frame dictionaries
            method: Condensation method ('keyframe', 'temporal', 'adaptive')
            **kwargs: Additional arguments for specific methods
        
        Returns:
            List of condensed frames
        """
        if not frames:
            return []

        if method == 'keyframe':
            return self._condense_keyframe(frames, **kwargs)
        elif method == 'temporal':
            return self._condense_temporal(frames, **kwargs)
        elif method == 'adaptive':
            return self._condense_adaptive(frames, **kwargs)
        else:
            raise ValueError(f"Unsupported condensation method: {method}")

    def _condense_keyframe(self, frames: List[Dict], threshold_factor: float = 1.0) -> List[Dict]:
        """Condense frames by keeping only key frames with significant changes."""
        condensed = [frames[0]]  # Always keep first frame
        
        for frame in frames[1:]:
            if self._is_significant_change(frame, condensed[-1], threshold_factor):
                condensed.append(frame)
        
        return condensed

    def _condense_temporal(self, frames: List[Dict], window_size: float = 0.5) -> List[Dict]:
        """Condense frames by averaging within time windows."""
        current_window = []
        condensed = []
        current_time = None

        for frame in frames:
            timestamp = datetime.fromisoformat(frame['raw_timestamp'])
            
            if current_time is None:
                current_time = timestamp

            if (timestamp - current_time).total_seconds() <= window_size:
                current_window.append(frame)
            else:
                if current_window:
                    condensed.append(self._average_window(current_window))
                current_window = [frame]
                current_time = timestamp

        if current_window:
            condensed.append(self._average_window(current_window))

        return condensed

    def _condense_adaptive(self, frames: List[Dict], target_count: int = None) -> List[Dict]:
        """Adaptively condense frames based on content complexity."""
        if target_count is None:
            target_count = max(3, len(frames) // 3)

        frame_scores = [(i, self._calculate_frame_importance(frame)) 
                       for i, frame in enumerate(frames)]
        
        # Sort by importance but keep some temporal distribution
        frame_scores.sort(key=lambda x: x[1], reverse=True)
        selected_indices = [x[0] for x in frame_scores[:target_count]]
        selected_indices.sort()  # Restore temporal order
        
        return [frames[i] for i in selected_indices]

    def _is_significant_change(self, frame1: Dict, frame2: Dict, threshold_factor: float) -> bool:
        """Check if there's a significant change between frames."""
        metrics = [
            ('derived_visibility', self.thresholds['visibility']),
            ('derived_distance_to_ego', self.thresholds['distance']),
            ('raw_num_lidar_points', self.thresholds['lidar_points'])
        ]

        for metric, threshold in metrics:
            if abs(frame1[metric] - frame2[metric]) > threshold * threshold_factor:
                return True
        return False

    def _average_window(self, window_frames: List[Dict]) -> Dict:
        """Average all frames in a time window."""
        if not window_frames:
            return None

        avg_frame = window_frames[0].copy()
        num_frames = len(window_frames)

        # Average numerical fields
        numeric_fields = [
            'derived_visibility',
            'derived_distance_to_ego',
            'raw_num_lidar_points'
        ]

        for field in numeric_fields:
            values = [frame[field] for frame in window_frames]
            avg_frame[field] = sum(values) / num_frames

        # Average arrays
        array_fields = ['raw_location', 'raw_velocity']
        for field in array_fields:
            if field in avg_frame:
                arrays = [frame[field] for frame in window_frames]
                avg_frame[field] = (np.array(arrays).mean(axis=0)).tolist()

        # Use middle frame's timestamp
        mid_idx = num_frames // 2
        avg_frame['raw_timestamp'] = window_frames[mid_idx]['raw_timestamp']

        return avg_frame

    def _calculate_frame_importance(self, frame: Dict) -> float:
        """Calculate frame importance based on multiple factors."""
        weights = {
            'visibility': 0.4,
            'lidar_points': 0.4,
            'radar_points': 0.2
        }

        score = (
            weights['visibility'] * frame['derived_visibility'] +
            weights['lidar_points'] * min(1.0, frame['raw_num_lidar_points'] / 50) +
            weights['radar_points'] * min(1.0, frame.get('raw_num_radar_points', 0) / 10)
        )

        return score