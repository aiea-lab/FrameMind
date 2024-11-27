from typing import List, Dict
import numpy as np
from .base_condenser import BaseCondenser, CondenserConfig

class SceneCondenser(BaseCondenser):
    def __init__(self, config: CondenserConfig):
        super().__init__(config)
    
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        if not frames:
            return []
        
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
        """Check if two scene frames can be merged"""
        # Check time window
        time_diff = self.calculate_time_difference(
            frame1['timestamp'],
            frame2['timestamp']
        )
        if time_diff > self.config.time_window:
            return False
        
        # Check object count
        if (len(frame1.get('objects', [])) > self.config.max_scene_objects or
            len(frame2.get('objects', [])) > self.config.max_scene_objects):
            return False
        
        return True
    
    def _merge_scene_frames(self, frames: List[Dict]) -> Dict:
        """Merge scene frames while preserving relationships"""
        if not frames:
            return None
        
        # Use frame with most complete data as base
        base_frame = max(frames, 
                        key=lambda x: len(x.get('objects', [])))
        merged = base_frame.copy()
        
        # Merge object lists while removing duplicates
        all_objects = set()
        for frame in frames:
            all_objects.update(frame.get('objects', []))
        merged['objects'] = list(all_objects)
        
        # Update scene metrics
        merged['scene_metrics'] = self._merge_scene_metrics(frames)
        
        return merged
    
    def _merge_scene_metrics(self, frames: List[Dict]) -> Dict:
        """Merge scene-level metrics"""
        metrics = {}
        
        # Calculate average metrics
        metrics['object_count'] = np.mean([len(f.get('objects', [])) 
                                         for f in frames])
        metrics['scene_coverage'] = np.mean([f.get('scene_coverage', 0) 
                                           for f in frames])
        metrics['frame_count'] = len(frames)
        
        return metrics