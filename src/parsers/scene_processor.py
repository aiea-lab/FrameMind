import os
import numpy as np
from datetime import datetime
from nuscenes import NuScenes
from src.core.coordinate import Coordinate
from src.core.frame_manager import FrameManager
from src.core.status import Status
from src.parsers.nuscenes_parser_config import NuScenesDataParserConfig
from src.condensing.static_condenser import StaticCondenser
import random
import json

def process_one_scene(nusc, scene_token):
    """Process one scene into sample, scene, and object frames."""
    # Retrieve scene metadata
    scene = nusc.get('scene', scene_token)
    scene_name = scene['name']
    print(f"Processing scene: {scene_name}")
    sample_token = scene['first_sample_token']
    
    # Retrieve start and end timestamps from samples
    start_sample = nusc.get('sample', scene['first_sample_token'])
    end_sample = nusc.get('sample', scene['last_sample_token'])
    start_time = start_sample['timestamp']
    end_time = end_sample['timestamp']
    
    # Convert timestamps to ISO format
    start_time_iso = datetime.fromtimestamp(start_time / 1e6).isoformat()
    end_time_iso = datetime.fromtimestamp(end_time / 1e6).isoformat()

    # Containers for outputs
    sample_frames = []
    object_annotations = {}

    # Process all samples in the scene
    while sample_token:
        sample = nusc.get('sample', sample_token)
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
        
        # Create Sample Frame
        sample_frame = {
            "sample_id": sample['token'],
            "timestamp": datetime.fromtimestamp(sample['timestamp'] / 1e6).isoformat(),
            "ego_vehicle": {
                "location": ego_pose['translation'],
                "orientation": ego_pose['rotation']
            },
            "sensor_data": {
                "lidar_path": nusc.get_sample_data_path(sample['data']['LIDAR_TOP']),
                "camera_paths": {
                    cam: nusc.get_sample_data_path(sample['data'][cam])
                    for cam in sample['data'] if "CAM" in cam
                }
            },
            "annotations": []
        }
        
        # Process Annotations
        for ann_token in sample['anns']:
            annotation = nusc.get('sample_annotation', ann_token)
            obj_id = annotation['instance_token']
            base_size = annotation['size']
            obj_data = {
                "object_id": obj_id,
                "category": annotation['category_name'],
                "location": annotation['translation'],
                "static_size": base_size,  # Object-level size
                "rotation": annotation['rotation']  # Object-level rotation
            }
            sample_frame['annotations'].append(obj_data)
            
            # Collect Object Data for Aggregation
            if obj_id not in object_annotations:
                object_annotations[obj_id] = {
                    "object_id": obj_id,
                    "category": annotation['category_name'],
                    "static_size": base_size,  # Static size
                    "rotation": annotation['rotation'],  # Static rotation
                    "motion_metadata": [],
                    "num_lidar_points": 0,
                    "num_radar_points": 0,
                    "relationships": {
                        "ego_vehicle_distance": None,
                        "neighboring_objects": []
                    }
                }
            
            # Corrected Velocity Handling
            velocity = nusc.box_velocity(ann_token)
            if velocity is None:
                velocity = [0.0, 0.0, 0.0]  # Default velocity
            else:
                velocity = velocity.tolist() if isinstance(velocity, np.ndarray) else velocity

            # Calculate distance to ego vehicle
            ego_vehicle_loc = sample_frame["ego_vehicle"]["location"]
            distance_to_ego = sum(
                (ego_vehicle_loc[i] - annotation['translation'][i])**2
                for i in range(3)
            )**0.5

            # Update Motion Metadata with all fields
            dynamic_size = [
                dim + random.uniform(-0.1, 0.1) for dim in base_size
            ]
            object_annotations[obj_id]["motion_metadata"].append({
                "timestamp": sample_frame['timestamp'],
                "location": annotation['translation'],
                "dynamic_size": dynamic_size,  # Time-varying size
                "velocity": velocity,
                "acceleration": [random.uniform(-0.5, 0.5) for _ in range(3)],  # Placeholder
                "angular_velocity": [random.uniform(-0.1, 0.1) for _ in range(3)],  # Placeholder
                "angular_acceleration": [random.uniform(-0.01, 0.01) for _ in range(3)],  # Placeholder
                "distance_to_ego": distance_to_ego,
                "num_lidar_points": annotation['num_lidar_pts'],
                "num_radar_points": annotation['num_radar_pts'],
                "camera_data": {
                    "bounding_boxes": {
                        cam: [random.randint(0, 100), random.randint(100, 200), random.randint(200, 300), random.randint(300, 400)]
                        for cam in sample['data'] if "CAM" in cam
                    },
                    "segmentation_masks": {
                        cam: f"path/to/mask_{cam.lower()}.png"
                        for cam in sample['data'] if "CAM" in cam
                    }
                },
                "occlusion_status": "Partially Occluded" if random.random() > 0.5 else "Visible",
                "visibility": random.uniform(0.0, 1.0),  # Visibility score
                "trajectory_history": [],  # Placeholder for trajectory history
                "predicted_trajectory": [],  # Placeholder for predicted trajectory
                "interaction_with_ego": {
                    "relative_position": [
                        annotation['translation'][i] - ego_vehicle_loc[i]
                        for i in range(3)
                    ],
                    "potential_collision": random.random() > 0.9
                },
                "sensor_fusion_confidence": random.uniform(0.8, 1.0)  # High-confidence score
            })
            
            # Update Aggregated Fields
            object_annotations[obj_id]["num_lidar_points"] += annotation['num_lidar_pts']
            object_annotations[obj_id]["num_radar_points"] += annotation['num_radar_pts']
        
        sample_frames.append(sample_frame)
        sample_token = sample['next']
    
    # Create Scene Frame
    scene_frame = {
        "scene_id": scene['token'],
        "scene_name": scene['name'],
        "duration": {
            "start": start_time_iso,
            "end": end_time_iso
        },
        "samples": sample_frames
    }
    
    # Create Object Frames
    object_frames = []
    for obj_id, obj_data in object_annotations.items():
        # Append object data to frames
        object_frames.append(obj_data)

    # Save object frames to output_frames.json
    # output_dir = "output"
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = os.path.join(output_dir, "output_frames.json")
    # with open(output_file, "w") as f:
    #     json.dump(object_frames, f, indent=4)
    # print(f"Object frames saved to {output_file}")

    # Return Results
    return {
        "scene_frame": scene_frame,
        "sample_frames": sample_frames,
        "object_frames": object_frames
    }

def process_scene(nusc: NuScenes, config: NuScenesDataParserConfig, frame_manager: FrameManager, scene_token: str):
    """Process one scene and create frames from samples"""
    scene = nusc.get('scene', scene_token)
    sample_token = scene['first_sample_token']
    prev_frame = None
    
    while sample_token:
        sample = nusc.get('sample', sample_token)
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        # Extract necessary information
        timestamp = datetime.fromtimestamp(sample['timestamp'] / 1e6)
        coordinates = Coordinate(ego_pose['translation'][0], ego_pose['translation'][1], ego_pose['translation'][2])

        # Create frame
        print(f"Creating frame: {sample['token']} at timestamp {timestamp} with coordinates {coordinates}")
        frame = frame_manager.create_frame(
            name=sample['token'],
            timestamp=timestamp,
            status=Status.ACTIVE, 
            coordinates=coordinates,
            elements=[]
        )

        # Add LIDAR and camera data
        lidar_path = nusc.get_sample_data_path(sample['data']['LIDAR_TOP'])
        frame.add_slot('lidar_path', lidar_path)

        for cam in config.cameras:
            if cam in sample['data']:
                cam_path = nusc.get_sample_data_path(sample['data'][cam])
                frame.add_slot(f'{cam.lower()}_path', cam_path)

        # Add annotations
        for ann_token in sample['anns']:
            ann = nusc.get('sample_annotation', ann_token)
            category = ann['category_name']
            location = ann['translation']
            frame.add_slot(f'annotation_{ann_token}', {
                'category': category,
                'location': Coordinate(location[0], location[1], location[2]),
                'size': ann['size'],
                'rotation': ann['rotation'],
                'velocity': nusc.box_velocity(ann_token),
                'num_lidar_pts': ann['num_lidar_pts'],
                'num_radar_pts': ann['num_radar_pts']
            })

        # Add neighbors
        if prev_frame:
            frame_manager.add_neighbors(prev_frame.name, frame.name)
            print(f"Adding neighbor: {prev_frame.name} is now a neighbor of {frame.name}")

        prev_frame = frame
        sample_token = sample['next']