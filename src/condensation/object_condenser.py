from typing import List, Dict, Tuple
import numpy as np
from sklearn.cluster import DBSCAN
import torch
import torch.nn as nn
from .base_condenser import BaseCondenser, CondenserConfig

class ObjectCondenser(BaseCondenser):
    def __init__(self, config: CondenserConfig):
        super().__init__(config)
        self.pattern_detector = DBSCAN(eps=0.3, min_samples=5)
    
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        if not frames:
            return []
        
        # Detect motion patterns
        patterns = self._detect_motion_patterns(frames)
        
        condensed_frames = []
        current_group = []
        current_pattern = patterns[0]
        
        for i, frame in enumerate(frames):
            if patterns[i] == current_pattern and len(current_group) < 5:
                current_group.append(frame)
            else:
                if current_group:
                    merged = self._merge_object_frames(current_group)
                    condensed_frames.append(merged)
                current_group = [frame]
                current_pattern = patterns[i]
        
        if current_group:
            merged = self._merge_object_frames(current_group)
            condensed_frames.append(merged)
        
        return condensed_frames
    
    def _detect_motion_patterns(self, frames: List[Dict]) -> np.ndarray:
        """Detect motion patterns using DBSCAN"""
        features = []
        for frame in frames:
            metadata = frame['motion_metadata'][0]
            feature = np.concatenate([
                metadata['velocity'],
                metadata['location'],
                [metadata.get('sensor_fusion_confidence', 0)]
            ])
            features.append(feature)
        
        return self.pattern_detector.fit_predict(np.array(features))
    
    def _merge_object_frames(self, frames: List[Dict]) -> Dict:
        """Merge object frames with motion averaging"""
        if not frames:
            return None
        
        base_frame = max(frames, 
                        key=lambda x: x['motion_metadata'][0]['sensor_fusion_confidence'])
        merged = base_frame.copy()
        metadata = merged['motion_metadata'][0]
        
        # Average motion data
        all_metadata = [f['motion_metadata'][0] for f in frames]
        metadata['location'] = np.mean([m['location'] for m in all_metadata], 
                                     axis=0).tolist()
        metadata['velocity'] = np.mean([m['velocity'] for m in all_metadata], 
                                     axis=0).tolist()
        
        return merged