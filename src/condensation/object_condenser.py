import numpy as np
from typing import List, Dict
from .base_condenser import BaseCondenser
from .config import CondensationConfig

class ObjectCondenser(BaseCondenser):
    """Condenser for object frames"""
    
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        if not frames:
            return []

        # Validate frames
        for frame in frames:
            if 'motion_metadata' not in frame or not frame['motion_metadata']:
                raise ValueError(f"Frame is missing 'motion_metadata': {frame}")

        condensed_frames = []
        current_group = []

        for frame in frames:
            if not current_group:
                current_group.append(frame)
                continue

            try:
                if self._can_merge_frames(current_group[-1], frame):
                    current_group.append(frame)
                else:
                    merged = self._merge_object_frames(current_group)
                    condensed_frames.append(merged)
                    current_group = [frame]
            except KeyError as e:
                print(f"Skipping frame due to missing key: {e}, frame: {frame}")

        if current_group:
            merged = self._merge_object_frames(current_group)
            condensed_frames.append(merged)

        return condensed_frames

    def _can_merge_frames(self, frame1: Dict, frame2: Dict) -> bool:
        """Check if frames can be merged"""
        try:
            meta1 = frame1['motion_metadata'][0]
            meta2 = frame2['motion_metadata'][0]

            # Time check
            time_diff = self._calculate_time_difference(
                meta1['raw_timestamp'],
                meta2['raw_timestamp']
            )
            if time_diff > self.config.time_window:
                return False

            # Position check
            pos1 = np.array(meta1['raw_location'])
            pos2 = np.array(meta2['raw_location'])
            if np.linalg.norm(pos2 - pos1) > self.config.max_position_gap:
                return False

            # Confidence check
            if (meta1.get('sensor_fusion_confidence', 0) < self.config.min_confidence or
                    meta2.get('sensor_fusion_confidence', 0) < self.config.min_confidence):
                return False

            return True

        except KeyError as e:
            print(f"Missing key in frame metadata: {e}")
            raise
    def _merge_object_frames(self, frames: List[Dict]) -> Dict:
        """Merge object frames"""
        if not frames:
            return None

        # Use frame with highest confidence as base
        base_frame = max(frames, 
                        key=lambda x: x['motion_metadata'][0].get('sensor_fusion_confidence', 0))
        merged = base_frame.copy()
        motion_data = merged['motion_metadata'][0]

        # Calculate weights based on confidence
        confidences = [f['motion_metadata'][0].get('sensor_fusion_confidence', 0) 
                      for f in frames]
        weights = np.array(confidences)
        weights = weights / weights.sum()

        # Merge motion data
        all_meta = [f['motion_metadata'][0] for f in frames]
        motion_data['raw_location'] = np.average([m['raw_location'] for m in all_meta], 
                                           weights=weights, axis=0).tolist()
        
        if 'velocity' in motion_data:
            motion_data['velocity'] = np.average([m.get('velocity', [0,0,0]) for m in all_meta],
                                               weights=weights, axis=0).tolist()

        # Add condensation metadata
        motion_data['condensed_info'] = {
            'frame_count': len(frames),
            'time_span': self._calculate_time_difference(
                frames[0]['motion_metadata'][0]['raw_timestamp'],
                frames[-1]['motion_metadata'][0]['raw_timestamp']
            ),
            'average_confidence': float(np.mean(confidences))
        }

        return merged