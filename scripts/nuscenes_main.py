import sys
from pathlib import Path
import os
import json

# from logger import get_logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))


#Initialize logger
#logger = get_logger("nuscenes_parser.log", logging.INFO)  # Create a logger

sys.path.append('/Users/ananya/Desktop/frames/data/raw/nuscenes')

# logger.info("Checking nuscenes-devkit path.")

# if os.path.exists('/Users/ananya/Documents/frames/nuscenes-devkit'):
#     logger.info("Path exists.")
# else:
#     logger.error("Path does not exist!")

from nuscenes import NuScenes
from core.coordinate import Coordinate
from core.frame_manager import FrameManager
from core.status import Status
# from src.trajectory import Trajectory
# from src.condensing import StaticCondenser
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
from nuscenes.nuscenes import NuScenes
from scipy.spatial.transform import Rotation as R
from condensing.static_condenser import StaticCondenser
from core.coordinate import Coordinate
from parsers.numpy_encoder import NumpyEncoder
# from parsers.nuscenes_parser_config import NuScenesDataParserConfig
from parsers.scene_elements import Cameras, SceneBox
from elements.cameras import Cameras
from elements.scene_box import SceneBox
from elements.camera_data import CameraData
from elements.sample import Sample
from core.scene import Scene
from core.nuscenes_database import NuScenesDatabase


@dataclass
class DataparserOutputs:
    image_filenames: List[str]
    cameras: Cameras
    scene_box: SceneBox
    metadata: Dict

@dataclass
class NuScenesDataParserConfig:
    data: Path
    data_dir: Path
    version: str = "v1.0-mini"
    cameras: List[str] = field(default_factory=lambda: [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ])
    mask_dir: Optional[Path] = None
    train_split_fraction: float = 0.9
    verbose: bool = True


class NuScenesParser:
    def __init__(self, config: NuScenesDataParserConfig):
        self.config = config
        self.database = NuScenesDatabase(config)

    def parse_all_scenes(self):
        # Loop through all scenes in the dataset
        all_scene_outputs = []
        for scene in self.database.nusc.scene:
            scene_token = scene['token']
            scene_output = self.parse_scene(scene_token)
            all_scene_outputs.append(scene_output)
        return all_scene_outputs

    def parse_scene(self, scene_token: str):
        # Load and parse a single scene
        self.database.load_scene(scene_token)
        return self._generate_scene_outputs(scene_token)

    def _generate_scene_outputs(self, scene_token: str):
        # Get the scene object
        scene = self.database.scenes[scene_token]
        sample_outputs = []

        # Iterate over all samples (snapshots) in the scene
        for sample in scene.samples:
            sample_output = self._generate_sample_output(sample)
            sample_outputs.append(sample_output)

        # Return scene-level output containing all samples' data
        return {
            'scene_name': scene.name,
            'samples': sample_outputs
        }

    def _generate_sample_output(self, sample: Sample):
        image_filenames = []
        poses = []
        intrinsics = []
        annotations = []  # New list to store annotation details

        # Collect camera data for each camera in the sample
        for camera in self.config.cameras:
            camera_data = sample.camera_data.get(camera)
            if camera_data:
                image_filenames.append(camera_data.image_path)
                poses.append(camera_data.extrinsics)
                intrinsics.append(camera_data.intrinsics)

        poses = np.stack(poses, axis=0)
        intrinsics = np.stack(intrinsics, axis=0)

        # Process annotations
        for ann_token in sample.annotations:
            ann = self.database.nusc.get('sample_annotation', ann_token)
            annotation_details = {
                'category': ann['category_name'],
                'location': ann['translation'],
                'size': ann['size'],
                'rotation': ann['rotation'],
                'velocity': self.database.nusc.box_velocity(ann_token),
                'num_lidar_pts': ann['num_lidar_pts'],
                'num_radar_pts': ann['num_radar_pts']
                //description
            }
            annotations.append(annotation_details)

        # Define the bounding box (you can replace this with actual bounding box data if available)
        scene_box = {
            'aabb': [[-1, -1, -1], [1, 1, 1]],
            'near': 0.1,
            'far': 100.0,
            'radius': 1.0,
            'collider_type': 'box'
        }

        # Return the data for a single sample (snapshot)
        return {
            'timestamp': sample.timestamp,
            'image_filenames': image_filenames,
            'camera_poses': poses,
            'camera_intrinsics': intrinsics,
            'scene_box': scene_box,
            'annotations': annotations  # Include annotations in the output
        }

def create_nuscenes_frames(dataroot: str, version: str = 'v1.0-mini'):

    print("abc")

    # Initialize NuScenes
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    
    # Initialize FrameManager
    frame_manager = FrameManager()
    prev_frame=None

    # Process each scene
    for scene in nusc.scene:
        print(f"Processing scene: {scene['name']}")
        
        # Get the first sample in the scene
        sample_token = scene['first_sample_token']
        
        while sample_token:
            sample = nusc.get('sample', sample_token)
            
            # Get ego pose
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])
            
            # Define elements
            elements = [] # Initialize elements, adjust this based on what elements should contain

            # Create a frame
            timestamp = datetime.fromtimestamp(sample['timestamp'] / 1e6)  # nuScenes timestamps are in microseconds
            coordinate = Coordinate(ego_pose['translation'][0], ego_pose['translation'][1], ego_pose['translation'][2])
            
            print(f"Creating frame: {sample['token']}")
            frame = frame_manager.create_frame(
                name=sample['token'],
                timestamp=timestamp,
                status=Status.ACTIVE,
                coordinates=coordinate,
                elements= elements
            )

            # Check if neighbor_frames attribute exists
            print(f"Created frame: {frame}, Neighbors attribute: {hasattr(frame, 'neighbor_frames')}")

            # Add LIDAR data
            lidar_path = nusc.get_sample_data_path(sample['data']['LIDAR_TOP'])
            frame.add_slot('lidar_path', lidar_path)

            # Add camera data
            for cam in ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT']:
                if cam in sample['data']:
                    cam_path = nusc.get_sample_data_path(sample['data'][cam])
                    frame.add_slot(f'{cam.lower()}_path', cam_path)

            # Process annotations
            for ann_token in sample['anns']:
                ann = nusc.get('sample_annotation', ann_token)
                category = ann['category_name']
                location = ann['translation']
                # Define annotation_data here
                annotation_data = {
                    'category': category,
                    'location': Coordinate(location[0], location[1], location[2]),
                    'size': ann['size'],
                    'rotation': ann['rotation'],
                    'velocity': nusc.box_velocity(ann_token),
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts']
                }
                frame.add_slot(f'annotation_{ann_token}', {
                    'category': category,
                    'location': Coordinate(location[0], location[1], location[2]),
                    'size': ann['size'],
                    'rotation': ann['rotation'],
                    'velocity': nusc.box_velocity(ann_token),
                    'num_lidar_pts': ann['num_lidar_pts'],
                    'num_radar_pts': ann['num_radar_pts']
                })
            
            frame.add_nuscenes_annotation(annotation_data)
            # If there's a previous frame, set them as neighbors
            if prev_frame:
                frame_manager.add_neighbors(prev_frame.name, frame.name)

            prev_frame = frame  # Update the previous frame for the next iteration
            sample_token = sample['next']

    return frame_manager

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

def process_and_condense_frames(dataroot: str, version: str = "v1.0-mini"):
    """Process frames, update trajectories, and apply condensation."""
    # Initialize NuScenes, FrameManager, and configuration
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    frame_manager = FrameManager()
    config = NuScenesDataParserConfig(
        data=Path(dataroot),
        data_dir=Path(dataroot),
        version=version
    )

    # Parse scenes
    parser = NuScenesParser(config)
    all_scene_outputs = parser.parse_all_scenes()

    # Update trajectories
    frame_manager.update_trajectories()

    # Example of getting a specific object trajectory
    object_id = "1ddf7a48af3d43be8763f154974f09ce"
    try:
        trajectory = frame_manager.get_object_trajectory(object_id)
        print(f"Trajectory for object {object_id}:", trajectory)
    except ValueError as e:
        print(e)

    # Condense frames
    condenser = StaticCondenser()  # Initialize the static condenser
    condensed_frames = condenser.condense_frames(frame_manager.frames)
    
    # Output condensed frames to JSON
    output_file_path = "condensed_frames_output.json"
    with open(output_file_path, 'w') as outfile:
        json.dump([frame.to_dict() for frame in condensed_frames], outfile, indent=4, cls=NumpyEncoder)

    print(f"Condensed frames have been written to {output_file_path}")

    return all_scene_outputs, condensed_frames

def main():

    dataroot = '/Applications/FrameMind/data/raw/nuscenes/v1.0-mini' # Update this path
    version = "v1.0-mini" 

    frame_manager = create_nuscenes_frames(dataroot)

    # Create the config
    config = NuScenesDataParserConfig(
        data=Path(dataroot),
        data_dir=Path(dataroot),
        version="v1.0-mini"
    )

    # Initialize NuScenes to get a scene token
    nusc = NuScenes(version=config.version, dataroot=str(config.data_dir), verbose=config.verbose)

    # Create and use the parser
    parser = NuScenesParser(config)  # Only config is passed now
    all_scene_outputs = parser.parse_all_scenes()
    # Process frames, update trajectories, and apply condensation
    # all_scene_outputs, condensed_frames = process_and_condense_frames(dataroot, version)

    # Retrieve frames from frame_manager
    frame_manager = create_nuscenes_frames(dataroot, version)
    frames = frame_manager.frames  # Assuming frame_manager.frames is a list of Frame objects

    distance_threshold = 5.0  # Example: 5 meters
    time_threshold = 2.0      # Example: 2 seconds  

    # Combine similar frames
    frame_manager.combine_similar_frames(distance_threshold, time_threshold)

    #Write the parsed data to a JSON file
    output_file_path = "parsed_scenes_output.json"
    with open(output_file_path, 'w') as outfile:
        json.dump(all_scene_outputs, outfile, indent=4, cls=NumpyEncoder)

    print(f"Output has been written to {output_file_path}")

    # Simulate frame communication for the current scene
    print("\nSimulating frame communication for one scene:")
    frame_manager.simulate_communication()

    # Output the results of the communication simulation
    print("\nFrames and their shared info after communication:")
    for frame in frame_manager.frames:
        print(f"Frame {frame.name}: Shared info = {frame.shared_info}")

    # Print outputs for each scene and sample
    for scene_output in all_scene_outputs:
        print(f"Scene: {scene_output['scene_name']}")
        for sample in scene_output['samples']:
            print(f"  Sample timestamp: {sample['timestamp']}")
            # print(f"  Number of images: {len(sample['image_filenames'])}")
            # print(f"  Camera poses shape: {sample['camera_poses'].shape}")
            # print(f"  Camera intrinsics shape: {sample['camera_intrinsics'].shape}")
            # print(f"  Scene box AABB: {sample['scene_box']['aabb']}")

    # Get the number of frames
    number_of_frames = frame_manager.get_number_of_frames()
    print(f"Number of frames: {number_of_frames}")

    #Static condensation

    #Condensed frames output


if __name__ == "__main__":
    main()
