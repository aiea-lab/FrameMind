from typing import List, Dict
import numpy as np
from .base_condenser import BaseCondenser

class SceneCondenser(BaseCondenser):
    """Condenser for scene frames"""
    
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        if not frames:
            return []
            
        # Debug print
        print("Scene frame keys:", frames[0].keys())

        condensed_frames = []
        current_group = []

        for frame in frames:
            if not current_group:
                current_group.append(frame)
                continue

            if self._can_merge_scenes(current_group[-1], frame):
                current_group.append(frame)
            else:
                merged = self._merge_scene_frames(current_group)
                condensed_frames.append(merged)
                current_group = [frame]

        if current_group:
            merged = self._merge_scene_frames(current_group)
            condensed_frames.append(merged)

        return condensed_frames

    def _can_merge_scenes(self, frame1: Dict, frame2: Dict) -> bool:
        """Check if scene frames can be merged"""
        # Get timestamps from appropriate field (modify based on your structure)
        time1 = frame1.get('sample_timestamp', frame1.get('timestamp', ''))
        time2 = frame2.get('sample_timestamp', frame2.get('timestamp', ''))
        
        if not time1 or not time2:
            return True  # If no timestamps, allow merging
            
        time_diff = self._calculate_time_difference(time1, time2)
        return time_diff <= self.config.time_window

    def _merge_scene_frames(self, frames: List[Dict]) -> Dict:
        """Merge scene frames"""
        if not frames:
            return None

        base_frame = frames[0].copy()
        
        try:
            # Merge object lists if present
            all_objects = set()
            for frame in frames:
                objects = frame.get('objects', frame.get('object_ids', []))
                all_objects.update(objects)
            
            base_frame['objects'] = list(all_objects)

            # Update scene metrics
            base_frame['scene_metrics'] = {
                'object_count': len(all_objects),
                'frame_count': len(frames)
            }

            # Add timestamp information if available
            timestamp_field = 'sample_timestamp' if 'sample_timestamp' in base_frame else 'timestamp'
            if any(timestamp_field in f for f in frames):
                timestamps = [f[timestamp_field] for f in frames if timestamp_field in f]
                if timestamps:
                    base_frame['scene_metrics']['timestamp_range'] = {
                        'start': min(timestamps),
                        'end': max(timestamps)
                    }

        except Exception as e:
            print(f"Error merging scene frames: {e}")
            print("Frame structure:", base_frame.keys())
            return base_frame

        return base_frame