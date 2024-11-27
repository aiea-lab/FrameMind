from typing import List, Dict
import numpy as np
from .base_condenser import BaseCondenser

class SampleCondenser(BaseCondenser):
    """Condenser for NuScenes sample frames"""
    
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        if not frames:
            return []

        condensed_frames = []
        current_group = []

        for frame in frames:
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

    def _can_merge_samples(self, frame1: Dict, frame2: Dict) -> bool:
        """Check if sample frames can be merged"""
        time_diff = self._calculate_time_difference(
            frame1['timestamp'],
            frame2['timestamp']
        )
        return time_diff <= self.config.time_window

    def _merge_sample_frames(self, frames: List[Dict]) -> Dict:
        """Merge sample frames based on NuScenes structure"""
        if not frames:
            return None

        base_frame = frames[0].copy()
        
        # Merge sensor data
        merged_sensor_data = self._merge_sensor_data([f['sensor_data'] for f in frames])
        base_frame['sensor_data'] = merged_sensor_data
        
        # Merge annotations
        merged_annotations = self._merge_annotations([f['annotations'] for f in frames])
        base_frame['annotations'] = merged_annotations

        # Update ego vehicle data (use most recent)
        base_frame['ego_vehicle'] = frames[-1]['ego_vehicle']
        
        # Add condensation metadata
        base_frame['condensed_info'] = {
            'frame_count': len(frames),
            'time_span': self._calculate_time_difference(
                frames[0]['timestamp'],
                frames[-1]['timestamp']
            ),
            'original_sample_ids': [f['sample_id'] for f in frames]
        }

        return base_frame

    def _merge_sensor_data(self, sensor_data_list: List[Dict]) -> Dict:
        """Merge sensor data from multiple frames"""
        merged_data = {}
        
        # Combine data from all sensors
        for sensor_type in ['LIDAR_TOP', 'RADAR_FRONT', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT',
                          'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']:
            points_list = []
            for sensor_data in sensor_data_list:
                if sensor_type in sensor_data:
                    points_list.extend(sensor_data[sensor_type].get('points', []))
            
            if points_list:
                merged_data[sensor_type] = {
                    'points': points_list,
                    'point_count': len(points_list)
                }

        # Merge camera data
        camera_data = {}
        for camera_type in ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                          'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
            bboxes = []
            for sensor_data in sensor_data_list:
                if camera_type in sensor_data:
                    bboxes.extend(sensor_data[camera_type].get('bounding_boxes', []))
            
            if bboxes:
                camera_data[camera_type] = {
                    'bounding_boxes': self._average_bboxes(bboxes)
                }
        
        merged_data['cameras'] = camera_data
        return merged_data

    def _merge_annotations(self, annotations_list: List[List[Dict]]) -> List[Dict]:
        """Merge annotations from multiple frames"""
        # Group annotations by object ID
        object_annotations = {}
        
        for frame_annotations in annotations_list:
            for annotation in frame_annotations:
                obj_id = annotation.get('object_id')
                if obj_id not in object_annotations:
                    object_annotations[obj_id] = []
                object_annotations[obj_id].append(annotation)

        # Merge annotations for each object
        merged_annotations = []
        for obj_id, obj_annots in object_annotations.items():
            if len(obj_annots) > 0:
                merged_annot = obj_annots[0].copy()  # Use first annotation as base
                
                # Average positions and other numerical values
                if len(obj_annots) > 1:
                    numeric_keys = ['location', 'size', 'velocity']
                    for key in numeric_keys:
                        if key in merged_annot:
                            values = [ann[key] for ann in obj_annots if key in ann]
                            merged_annot[key] = np.mean(values, axis=0).tolist()

                merged_annotations.append(merged_annot)

        return merged_annotations

    def _average_bboxes(self, bboxes: List[List[float]]) -> List[float]:
        """Average multiple bounding boxes"""
        if not bboxes:
            return []
        return np.mean(bboxes, axis=0).tolist()