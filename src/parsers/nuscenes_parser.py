import json
from pathlib import Path
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from nuscenes.nuscenes import NuScenes
from src.parsers.nuscenes_parser_config import NuScenesDataParserConfig
from src.core.nuscenes_database import NuScenesDatabase, Sample
from parsers.scene_elements import Cameras, SceneBox
from src.processing import NuScenesParser


# Custom JSON encoder to handle non-serializable objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)  
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)

@dataclass
class NuScenesDataParserConfig:
    data: Path
    data_dir: Path
    version: str = "v1.0-mini"
    cameras: List[str] = field(default_factory=lambda: [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", "CAM_BACK", 
        "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ])
    mask_dir: Optional[Path] = None
    train_split_fraction: float = 0.9
    verbose: bool = True

# Define data structures
class Cameras:
    def __init__(self, poses: np.ndarray, intrinsics: np.ndarray, image_filenames: List[str]):
        self.poses = poses
        self.intrinsics = intrinsics
        self.image_filenames = image_filenames

class SceneBox:
    def __init__(self, aabb: List[List[float]], near: float, far: float, radius: float, collider_type: str):
        self.aabb = aabb
        self.near = near
        self.far = far
        self.radius = radius
        self.collider_type = collider_type

@dataclass
class DataparserOutputs:
    image_filenames: List[str]
    cameras: Cameras
    scene_box: SceneBox
    metadata: Dict

class NuScenesParser:
    def __init__(self, config: NuScenesDataParserConfig):
        self.config = config
        self.nusc = NuScenes(version=config.version, dataroot=str(config.data_dir), verbose=config.verbose)

    def parse_all_scenes(self):
        # Loop through all scenes in the dataset
        all_scene_outputs = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_output = self.parse_scene(scene_token)
            all_scene_outputs.append(scene_output)
        return all_scene_outputs

    def parse_scene(self, scene_token: str):
        # Load and parse a single scene
        self.nusc.load_scene(scene_token)
        return self._generate_scene_outputs(scene_token)

    def _generate_scene_outputs(self, scene_token: str):
        # Get the scene object
        scene = self.nusc.get_scene(scene_token)
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

    def _generate_sample_output(self, sample):
        # Example function to generate the output for a sample
        image_filenames = []
        poses = []
        intrinsics = []
        annotations = []

        # Define other camera processing code here
        # Define other annotation processing code here

        # Return data
        return {
            'timestamp': sample.timestamp,
            'image_filenames': image_filenames,
            'camera_poses': poses,
            'camera_intrinsics': intrinsics,
            'annotations': annotations
        }
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
                #description
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