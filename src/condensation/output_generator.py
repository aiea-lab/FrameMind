from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import numpy as np
from datetime import datetime
import json
from nuscenes import NuScenes

@dataclass
class CondensationParams:
    time_window: float = 0.2
    min_confidence: float = 0.3
    max_position_gap: float = 2.0

class CondensedOutputGenerator:
    def __init__(self, params: CondensationParams, nusc=None, scene_token=None):
        self.params = params
        self.nusc = nusc
        self.scene_token = scene_token

    def generate_condensed_output(self, original_frames: List[Dict], condensed_frames: List[Dict]) -> Dict:
        """Generate structured condensed output"""
        return {
            "meta_info": {
                "scene_token": self.scene_token if self.scene_token else "",
                "timestamp": datetime.now().isoformat(),
                "num_original_frames": len(original_frames),
                "num_condensed_frames": len(condensed_frames),
                "condensation_params": {
                    "time_window": self.params.time_window,
                    "min_confidence": self.params.min_confidence,
                    "max_position_gap": self.params.max_position_gap
                }
            },
            "ego_vehicle": self._extract_ego_state(original_frames),
            "condensed_frames": self._process_condensed_frames(condensed_frames),
            "statistics": self._calculate_statistics(original_frames, condensed_frames)
        }

    def _generate_meta_info(self, original_frames: List[Dict], condensed_frames: List[Dict]) -> Dict:
        """Generate meta information with proper scene details"""
        scene_info = {}
        if self.nusc and self.scene_token:
            try:
                scene = self.nusc.get('scene', self.scene_token)
                scene_info = {
                    'scene_token': self.scene_token,
                    'scene_name': scene.get('name', ''),
                    'scene_description': scene.get('description', ''),
                    'log_token': scene.get('log_token', ''),
                    'nbr_samples': scene.get('nbr_samples', 0),
                    'first_sample_token': scene.get('first_sample_token', ''),
                    'last_sample_token': scene.get('last_sample_token', '')
                }

                # Get location from log if available
                if 'log_token' in scene:
                    log = self.nusc.get('log', scene['log_token'])
                    scene_info['location'] = log.get('location', 'unknown')
                    scene_info['date_captured'] = log.get('date_captured', '')
                    
            except Exception as e:
                print(f"Warning: Error extracting scene info: {e}")
                scene_info = {
                    'scene_token': self.scene_token,
                    'error': 'Failed to extract complete scene information'
                }
        
        return {
            **scene_info,  # Include scene information if available
            "timestamp": datetime.now().isoformat(),
            "num_original_frames": len(original_frames),
            "num_condensed_frames": len(condensed_frames),
            "condensation_params": {
                "time_window": self.params.time_window,
                "min_confidence": self.params.min_confidence,
                "max_position_gap": self.params.max_position_gap
            }
        }

    def _extract_ego_state(self, frames: List[Dict]) -> Dict:
        """Extract ego vehicle state from NuScenes frames"""
        if not frames:
            return self._get_default_ego_state()

        try:
            # Get the latest frame
            latest_frame = frames[-1]
            
            # Try to get ego pose from sample token
            if self.nusc and 'sample_token' in latest_frame:
                try:
                    # Get sample
                    sample = self.nusc.get('sample', latest_frame['sample_token'])
                    # Get ego pose from the first frame's LIDAR data
                    sd_record = self.nusc.get('sample_data', sample['data']['LIDAR_TOP'])
                    ego_pose = self.nusc.get('ego_pose', sd_record['ego_pose_token'])
                    
                    # Extract timestamp from motion metadata if available
                    timestamp = ""
                    if latest_frame.get('motion_metadata'):
                        timestamp = latest_frame['motion_metadata'][0].get('raw_timestamp', '')
                    
                    return {
                        "position": ego_pose['translation'],
                        "heading": float(np.arctan2(2.0 * (ego_pose['rotation'][1] * ego_pose['rotation'][2] + 
                                                        ego_pose['rotation'][3] * ego_pose['rotation'][0]), 
                                                1.0 - 2.0 * (ego_pose['rotation'][2]**2 + 
                                                            ego_pose['rotation'][3]**2))),
                        "velocity": ego_pose.get('velocity', [0.0, 0.0, 0.0]),
                        "timestamp": timestamp or str(ego_pose.get('timestamp', '')),
                        "source": "nuscenes_ego_pose",
                        "rotation": ego_pose['rotation'],  # Include full rotation
                        "acceleration": ego_pose.get('acceleration', [0.0, 0.0, 0.0])
                    }
                except Exception as e:
                    print(f"Failed to get ego pose from sample: {e}")
            
            # Try getting from motion metadata
            motion_metadata = latest_frame.get('motion_metadata', [])
            if motion_metadata:
                metadata = motion_metadata[0]
                return {
                    "position": metadata.get('raw_ego_position', [0, 0, 0]),
                    "heading": float(metadata.get('raw_ego_rotation', [0, 0, 0, 1])[2]),
                    "velocity": metadata.get('raw_ego_velocity', [0.0, 0.0, 0.0]),
                    "timestamp": metadata.get('raw_timestamp', datetime.now().isoformat()),
                    "source": "motion_metadata"
                }

        except Exception as e:
            print(f"Error extracting ego state: {e}")
        
        return self._get_default_ego_state()

    
    def _get_default_ego_state(self):
        """Return default ego state"""
        return {
            "position": [0, 0, 0],
            "heading": 0.0,
            "velocity": [0.0, 0.0, 0.0],
            "timestamp": datetime.now().isoformat(),
            "source": "default_values"
        }
    def _process_condensed_frames(self, condensed_frames: List[Dict]) -> List[Dict]:
        """Process and structure condensed frames"""
        processed_frames = []
        
        for frame in condensed_frames:
            processed_frame = self._process_single_frame(frame)
            if processed_frame:
                processed_frames.append(processed_frame)
                
        return processed_frames

    def _process_single_frame(self, frame: Dict) -> Dict:
        """Process a single condensed frame"""
        motion_metadata = frame.get('motion_metadata', [{}])[0]
        
        return {
            "object_info": {
                "object_id": frame.get('raw_object_id', ''),
                "category": frame.get('raw_category', ''),
                "static_size": frame.get('raw_static_size', [0, 0, 0]),
                "first_observed": motion_metadata.get('raw_timestamp', ''),
                "last_observed": motion_metadata.get('raw_timestamp', '')
            },
            "tracking_info": {
                "positions": self._process_positions(frame),
                "motion_metrics": self._calculate_motion_metrics(frame)
            },
            "sensor_data": self._process_sensor_data(frame),
            "quality_metrics": self._calculate_quality_metrics(frame),
            "condensation_metrics": self._calculate_condensation_metrics(frame)
        }

    def _process_positions(self, frame: Dict) -> List[Dict]:
        """Process position information"""
        positions = []
        for metadata in frame.get('motion_metadata', []):
            positions.append({
                "timestamp": metadata.get('raw_timestamp', ''),
                "location": metadata.get('raw_location', [0, 0, 0]),
                "velocity": metadata.get('raw_velocity', [0, 0, 0]),
                "acceleration": metadata.get('derived_acceleration', [0, 0, 0])
            })
        return positions

    def _calculate_motion_metrics(self, frame: Dict) -> Dict:
        """Calculate motion metrics"""
        velocities = [meta.get('raw_velocity', [0, 0, 0]) 
                     for meta in frame.get('motion_metadata', [])]
        
        if velocities:
            avg_velocity = np.mean(velocities, axis=0)
            is_stationary = np.all(np.abs(avg_velocity) < 0.1)
        else:
            avg_velocity = [0, 0, 0]
            is_stationary = True

        return {
            "avg_velocity": avg_velocity.tolist() if isinstance(avg_velocity, np.ndarray) else avg_velocity,
            "is_stationary": is_stationary,
            "distance_traveled": self._calculate_distance_traveled(frame)
        }

    def _process_sensor_data(self, frame: Dict) -> Dict:
        """Process sensor data information"""
        metadata_list = frame.get('motion_metadata', [])
        
        return {
            "lidar_info": {
                "avg_num_points": np.mean([meta.get('raw_num_lidar_points', 0) 
                                         for meta in metadata_list]),
                "point_range": [
                    min([meta.get('raw_num_lidar_points', 0) for meta in metadata_list]),
                    max([meta.get('raw_num_lidar_points', 0) for meta in metadata_list])
                ],
                "confidence": np.mean([meta.get('derived_sensor_fusion_confidence', 0) 
                                     for meta in metadata_list])
            },
            "camera_info": {
                "visibility": np.mean([meta.get('derived_visibility', 0) 
                                     for meta in metadata_list]),
                "occlusion_status": self._get_most_common_occlusion(metadata_list),
                "bounding_boxes": metadata_list[0].get('raw_camera_data', {}).get('raw_bounding_boxes', {})
                                if metadata_list else {}
            }
        }

    def _calculate_quality_metrics(self, frame: Dict) -> Dict:
        """Calculate quality metrics"""
        metadata_list = frame.get('motion_metadata', [])
        
        return {
            "tracking_confidence": np.mean([meta.get('derived_sensor_fusion_confidence', 0) 
                                          for meta in metadata_list]),
            "classification_confidence": frame.get('classification_confidence', 0.0),
            "avg_distance_to_ego": np.mean([meta.get('derived_distance_to_ego', 0) 
                                          for meta in metadata_list]),
            "sensor_fusion_confidence": np.mean([meta.get('derived_sensor_fusion_confidence', 0) 
                                               for meta in metadata_list])
        }

    def _calculate_condensation_metrics(self, frame: Dict) -> Dict:
        """Calculate condensation specific metrics"""
        metadata_list = frame.get('motion_metadata', [])
        positions = [meta.get('raw_location', [0, 0, 0]) for meta in metadata_list]
        velocities = [meta.get('raw_velocity', [0, 0, 0]) for meta in metadata_list]
        
        return {
            "num_frames_condensed": len(metadata_list),
            "time_span": self._calculate_time_span(metadata_list),
            "position_variance": float(np.var(positions).mean()) if positions else 0,
            "velocity_variance": float(np.var(velocities).mean()) if velocities else 0
        }

    def _calculate_statistics(self, original_frames: List[Dict], condensed_frames: List[Dict]) -> Dict:
        """Calculate overall statistics"""
        return {
            "frame_reduction_ratio": 1 - (len(condensed_frames) / len(original_frames)) if original_frames else 0,
            "avg_tracking_confidence": np.mean([
                frame.get('quality_metrics', {}).get('tracking_confidence', 0)
                for frame in condensed_frames
            ]),
            "avg_classification_confidence": np.mean([
                frame.get('quality_metrics', {}).get('classification_confidence', 0)
                for frame in condensed_frames
            ]),
            "processing_time": 0.15  # This could be actually measured
        }

    @staticmethod
    def _get_most_common_occlusion(metadata_list: List[Dict]) -> str:
        """Get most common occlusion status"""
        if not metadata_list:
            return "unknown"
            
        occlusion_states = [meta.get('derived_occlusion_status', 'unknown') 
                           for meta in metadata_list]
        return max(set(occlusion_states), key=occlusion_states.count)

    @staticmethod
    def _calculate_time_span(metadata_list: List[Dict]) -> float:
        """Calculate time span of frames"""
        if not metadata_list:
            return 0.0
            
        timestamps = [datetime.fromisoformat(meta['raw_timestamp']) 
                     for meta in metadata_list 
                     if 'raw_timestamp' in meta]
        
        if not timestamps:
            return 0.0
            
        return (max(timestamps) - min(timestamps)).total_seconds()

    @staticmethod
    def _calculate_distance_traveled(frame: Dict) -> float:
        """Calculate total distance traveled"""
        positions = [meta.get('raw_location', [0, 0, 0]) 
                    for meta in frame.get('motion_metadata', [])]
        
        if len(positions) < 2:
            return 0.0
            
        distances = np.diff(positions, axis=0)
        return float(np.sum(np.sqrt(np.sum(distances**2, axis=1))))

