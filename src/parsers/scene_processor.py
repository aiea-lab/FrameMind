import os
import numpy as np
from pyquaternion import Quaternion
from datetime import datetime
from nuscenes import NuScenes
from src.core.coordinate import Coordinate
from src.core.frame_manager import FrameManager
from src.core.status import Status
from src.parsers.nuscenes_parser_config import NuScenesDataParserConfig
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.data_classes import Box
# from src.condensing.static_condenser import StaticCondenser
import random
import json

def process_one_scene(nusc, scene_token):
    """Process one scene into sample, scene, and object frames with raw/derived data separation."""
    
    # Get raw scene data
    raw_scene = nusc.get('scene', scene_token)
    raw_scene_name = raw_scene['name']
    print(f"Processing scene: {raw_scene_name}")
    raw_sample_token = raw_scene['first_sample_token']
    
    # Get raw timestamp data
    raw_start_sample = nusc.get('sample', raw_scene['first_sample_token'])
    raw_end_sample = nusc.get('sample', raw_scene['last_sample_token'])
    raw_start_time = raw_start_sample['timestamp']
    raw_end_time = raw_end_sample['timestamp']
    
    # Derive ISO format timestamps
    derived_start_time_iso = datetime.fromtimestamp(raw_start_time / 1e6).isoformat()
    derived_end_time_iso = datetime.fromtimestamp(raw_end_time / 1e6).isoformat()

    # Initialize containers
    sample_frames = []
    object_annotations = {}

    # Process all samples in scene
    while raw_sample_token:
        # Get raw sample data
        raw_sample = nusc.get('sample', raw_sample_token)
        raw_lidar_data = nusc.get('sample_data', raw_sample['data']['LIDAR_TOP'])
        raw_ego_pose = nusc.get('ego_pose', raw_lidar_data['ego_pose_token'])
        
        # Create Sample Frame
        sample_frame = {
            "raw_sample_id": raw_sample['token'],
            "raw_timestamp": datetime.fromtimestamp(raw_sample['timestamp'] / 1e6).isoformat(),
            "raw_ego_vehicle": {
                "raw_location": raw_ego_pose['translation'],
                "raw_orientation": raw_ego_pose['rotation']
            },
            "raw_sensor_data": {
                "raw_lidar_path": nusc.get_sample_data_path(raw_sample['data']['LIDAR_TOP']),
                "raw_camera_paths": {
                    cam: nusc.get_sample_data_path(raw_sample['data'][cam])
                    for cam in raw_sample['data'] if "CAM" in cam
                }
            },
            "raw_annotations": []
        }
        
        # Process each annotation in the sample
        for raw_ann_token in raw_sample['anns']:
            # Get raw annotation data
            raw_annotation = nusc.get('sample_annotation', raw_ann_token)
            raw_obj_id = raw_annotation['instance_token']
            raw_base_size = raw_annotation['size']

            # Initialize object if not seen before
            if raw_obj_id not in object_annotations:
                object_annotations[raw_obj_id] = {
                    "raw_object_id": raw_obj_id,
                    "raw_category": raw_annotation['category_name'],
                    "raw_static_size": raw_base_size,
                    "motion_metadata": [],
                    "raw_num_lidar_points": 0,
                    "raw_num_radar_points": 0
                }

            # Get raw velocity data
            raw_velocity = nusc.box_velocity(raw_ann_token)
            if raw_velocity is None:
                raw_velocity = [0.0, 0.0, 0.0]
            raw_velocity = raw_velocity.tolist() if isinstance(raw_velocity, np.ndarray) else raw_velocity

            # Calculate derived ego vehicle distance
            raw_ego_location = sample_frame["raw_ego_vehicle"]["raw_location"]
            derived_distance_to_ego = np.linalg.norm(
                np.array(raw_ego_location) - np.array(raw_annotation['translation'])
            )

            # Create motion metadata with raw and derived data
            motion_metadata = {
                # Raw motion data
                "raw_timestamp": sample_frame['raw_timestamp'],
                "raw_location": raw_annotation['translation'],
                "raw_rotation": raw_annotation['rotation'],
                "raw_velocity": raw_velocity,
                "raw_num_lidar_points": raw_annotation['num_lidar_pts'],
                "raw_num_radar_points": raw_annotation['num_radar_pts'],
                "raw_camera_data": {
                    "raw_bounding_boxes": get_bounding_boxes(
                        nusc, 
                        raw_sample['data'], 
                        raw_annotation
                    )
                },

                # Derived motion data
                "derived_dynamic_size": [
                    dim + np.random.uniform(-0.05, 0.05) for dim in raw_base_size
                ],
                "derived_acceleration": calculate_acceleration(raw_velocity),
                "derived_angular_velocity": calculate_angular_velocity(raw_annotation['rotation']),
                "derived_angular_acceleration": calculate_angular_acceleration(),
                "derived_distance_to_ego": derived_distance_to_ego,
                "derived_occlusion_status": calculate_occlusion_status(raw_annotation),
                "derived_visibility": calculate_visibility(raw_annotation),
                "derived_interaction_with_ego": {
                    "derived_relative_position": np.subtract(
                        raw_annotation['translation'], 
                        raw_ego_location
                    ).tolist(),
                    "derived_potential_collision": calculate_collision_potential(
                        raw_annotation['translation'],
                        raw_ego_location,
                        raw_velocity
                    )
                },
                "derived_sensor_fusion_confidence": calculate_sensor_fusion_confidence(
                    raw_annotation['num_lidar_pts'],
                    raw_annotation['num_radar_pts']
                )
            }

            # Update object annotations
            object_annotations[raw_obj_id]["motion_metadata"].append(motion_metadata)
            object_annotations[raw_obj_id]["raw_num_lidar_points"] += raw_annotation['num_lidar_pts']
            object_annotations[raw_obj_id]["raw_num_radar_points"] += raw_annotation['num_radar_pts']

            # Add to sample frame annotations
            sample_frame['raw_annotations'].append({
                "raw_object_id": raw_obj_id,
                "raw_category": raw_annotation['category_name'],
                "raw_location": raw_annotation['translation'],
                "raw_rotation": raw_annotation['rotation']
            })

        # Add sample frame to list
        sample_frames.append(sample_frame)
        raw_sample_token = raw_sample['next']

    # Create scene frame
    scene_frame = {
        "raw_scene_id": raw_scene['token'],
        "raw_scene_name": raw_scene['name'],
        "derived_duration": {
            "derived_start": derived_start_time_iso,
            "derived_end": derived_end_time_iso
        },
        "raw_samples": sample_frames
    }

    # Convert object annotations to list
    object_frames = list(object_annotations.values())

    return {
        "scene_frame": scene_frame,
        "sample_frames": sample_frames,
        "object_frames": object_frames
    }

# Helper functions for derived calculations
def calculate_acceleration(velocity, prev_velocity=None):
    if prev_velocity is None:
        return [0.0, 0.0, 0.0]
    return np.subtract(velocity, prev_velocity).tolist()

def calculate_angular_velocity(rotation, prev_rotation=None):
    if prev_rotation is None:
        return [0.0, 0.0, 0.0]
    return np.subtract(rotation, prev_rotation).tolist()

def calculate_angular_acceleration():
    return [np.random.uniform(-0.01, 0.01) for _ in range(3)]

def calculate_occlusion_status(annotation):
    return "Partially Occluded" if annotation['num_lidar_pts'] < 10 else "Visible"

def calculate_visibility(annotation):
    return min(1.0, annotation['num_lidar_pts'] / 100)

def calculate_collision_potential(obj_pos, ego_pos, obj_velocity):
    distance = np.linalg.norm(np.subtract(obj_pos, ego_pos))
    return distance < 10.0 and np.linalg.norm(obj_velocity) > 1.0

def calculate_sensor_fusion_confidence(lidar_points, radar_points):
    return min(1.0, (lidar_points + radar_points) / 150)

def get_camera_box_coords(nusc, ann_token, camera_channel):
    """Get 2D bounding box coordinates for a given annotation in a camera view"""
    try:
        # Get sample and camera data
        annotation = nusc.get('sample_annotation', ann_token)
        sample = nusc.get('sample', annotation['sample_token'])
        cam_token = sample['data'][camera_channel]
        cam_data = nusc.get('sample_data', cam_token)
        
        # Get sensor calibration
        sensor = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
        intrinsic = np.array(sensor['camera_intrinsic'])
        
        # Get annotation box
        box = Box(
            center=annotation['translation'],
            size=annotation['size'],
            orientation=Quaternion(annotation['rotation'])
        )
        
        # Project box to image
        corners_3d = box.corners()
        corners_2d = view_points(corners_3d, intrinsic, normalize=True)
        
        # Calculate bounding box
        bbox = [
            float(max(0, min(corners_2d[0, :]))),          # x_min
            float(max(0, min(corners_2d[1, :]))),          # y_min
            float(min(cam_data['width'], max(corners_2d[0, :]))),   # x_max
            float(min(cam_data['height'], max(corners_2d[1, :])))   # y_max
        ]
        
        return bbox
    
    except Exception as e:
        print(f"Error getting camera box for {camera_channel}: {e}")
        return [0, 0, 0, 0]  # Default bbox
    
def get_bounding_boxes(nusc, sample_data, annotation):
    """Get 2D bounding boxes from annotations"""
    boxes = {}
    for cam in sample_data:
        if 'CAM' in cam:
            # Default/estimated bounding box
            boxes[cam] = [
                max(0, int(annotation.get('bbox_corners', [0])[0])),   # x_min
                max(0, int(annotation.get('bbox_corners', [0, 0])[1])), # y_min
                min(1600, int(annotation.get('bbox_corners', [0, 0, 800])[2])),  # x_max
                min(900, int(annotation.get('bbox_corners', [0, 0, 0, 450])[3]))  # y_max
            ]
    return boxes
