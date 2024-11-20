# src/core/nuscenes_database.py
from typing import Dict
from pathlib import Path
import numpy as np
from src.elements.camera_data import CameraData
from src.elements.sample import Sample
from src.core.scene import Scene  # Import the Scene class from scene.py
from nuscenes.nuscenes import NuScenes
from src.parsers.nuscenes_parser_config import NuScenesDataParserConfig
from scipy.spatial.transform import Rotation as R  # Ensure the path is correct


class NuScenesDatabase:
    def __init__(self, config: NuScenesDataParserConfig):  
        self.config = config
        self.nusc = NuScenes(version=config.version, dataroot=str(config.data_dir), verbose=config.verbose)
        self.scenes: Dict[str, Scene] = {}

    def load_data(self):
        scene = self.nusc.get('scene', scene_token)
        scene_obj = Scene(scene['token'], scene['name'])
        self.scenes[scene['token']] = scene_obj

        sample_token = scene['first_sample_token']
        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            sample_obj = Sample(sample['token'], sample['timestamp'], scene['token'])

            for camera in self.config.cameras:
                cam_data = self.nusc.get('sample_data', sample['data'][camera])
                calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                
                intrinsics = np.array(calib['camera_intrinsic'])
                extrinsics = np.eye(4)
                
                if 'rotation_matrix' in calib:
                    extrinsics[:3, :3] = np.array(calib['rotation_matrix'])
                elif 'rotation' in calib:
                    quaternion = np.array(calib['rotation'])
                    rotation_matrix = R.from_quat(quaternion).as_matrix()
                    extrinsics[:3, :3] = rotation_matrix
                else:
                    print(f"Warning: 'rotation' or 'rotation_matrix' not found for camera {camera}. Using identity matrix.")

                if 'translation' in calib:
                    extrinsics[:3, 3] = np.array(calib['translation'])
                else:
                    print(f"Warning: 'translation' not found for camera {camera}. Using zeros.")

                image_path = Path(self.nusc.get_sample_data_path(cam_data['token']))
                camera_data = CameraData(intrinsics, extrinsics, str(image_path))
                sample_obj.add_camera_data(camera, camera_data)

            scene_obj.add_sample(sample_obj)
            sample_token = sample['next']

    def load_scene(self, scene_token: str):
        # Load the scene using the provided scene_token
        scene = self.nusc.get('scene', scene_token)
        scene_obj = Scene(scene['token'], scene['name'])
        self.scenes[scene['token']] = scene_obj

        sample_token = scene['first_sample_token']
        while sample_token:
            sample = self.nusc.get('sample', sample_token)
            sample_obj = Sample(sample['token'], sample['timestamp'], scene['token'])

            # Store annotation tokens in the sample
            sample_obj.add_annotations(sample['anns'])

            for camera in self.config.cameras:
                cam_data = self.nusc.get('sample_data', sample['data'][camera])
                calib = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

                intrinsics = np.array(calib['camera_intrinsic'])
                extrinsics = np.eye(4)

                if 'rotation_matrix' in calib:
                    extrinsics[:3, :3] = np.array(calib['rotation_matrix'])
                elif 'rotation' in calib:
                    quaternion = np.array(calib['rotation'])
                    rotation_matrix = R.from_quat(quaternion).as_matrix()
                    extrinsics[:3, :3] = rotation_matrix
                else:
                    print(f"Warning: 'rotation' or 'rotation_matrix' not found for camera {camera}. Using identity matrix.")

                if 'translation' in calib:
                    extrinsics[:3, 3] = np.array(calib['translation'])
                else:
                    print(f"Warning: 'translation' not found for camera {camera}. Using zeros.")

                image_path = Path(self.nusc.get_sample_data_path(cam_data['token']))
                camera_data = CameraData(intrinsics, extrinsics, str(image_path))
                sample_obj.add_camera_data(camera, camera_data)

            scene_obj.add_sample(sample_obj)
            sample_token = sample['next']
