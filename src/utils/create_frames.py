from nuscenes import NuScenes
from src.core.frame_manager import FrameManager
from src.core.coordinate import Coordinate
from src.core.status import Status
from datetime import datetime

def create_nuscenes_frames(dataroot: str, version: str = 'v1.0-mini'):
    """Create frames from all scenes in the NuScenes dataset."""
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    frame_manager = FrameManager()
    prev_frame = None

    for scene in nusc.scene:
        print(f"Processing scene: {scene['name']}")
        sample_token = scene['first_sample_token']

        while sample_token:
            sample = nusc.get('sample', sample_token)
            lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
            ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

            # Create frame
            timestamp = datetime.fromtimestamp(sample['timestamp'] / 1e6)
            coordinates = Coordinate(*ego_pose['translation'])
            frame = frame_manager.create_frame(
                name=sample['token'],
                timestamp=timestamp,
                status=Status.ACTIVE,
                coordinates=coordinates,
                elements=[]
            )

            add_sensor_data_to_frame(frame, nusc, sample)
            add_annotations_to_frame(frame, nusc, sample)

            # Add neighbors
            if prev_frame:
                frame_manager.add_neighbors(prev_frame.name, frame.name)

            prev_frame = frame
            sample_token = sample['next']

    return frame_manager

def create_nuscenes_frame_for_scene(nusc, frame_manager, scene_token):
    """Create frames for a single scene."""
    scene = nusc.get('scene', scene_token)
    print(f"Processing scene: {scene['name']}")
    sample_token = scene['first_sample_token']
    prev_frame = None
    frames = []

    while sample_token:
        sample = nusc.get('sample', sample_token)
        lidar_data = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        ego_pose = nusc.get('ego_pose', lidar_data['ego_pose_token'])

        timestamp = datetime.fromtimestamp(sample['timestamp'] / 1e6)
        coordinates = Coordinate(*ego_pose['translation'])
        frame = frame_manager.create_frame(
            name=sample['token'],
            timestamp=timestamp,
            status=Status.ACTIVE,
            coordinates=coordinates,
            elements=[]
        )

        add_sensor_data_to_frame(frame, nusc, sample)
        add_annotations_to_frame(frame, nusc, sample)

        if prev_frame:
            frame_manager.add_neighbors(prev_frame.name, frame.name)

        frames.append(frame)
        prev_frame = frame
        sample_token = sample['next']

    return frames

def add_sensor_data_to_frame(frame, nusc, sample):
    """Add sensor data paths (lidar and camera) to a frame."""
    lidar_path = nusc.get_sample_data_path(sample['data']['LIDAR_TOP'])
    frame.add_slot('lidar_path', lidar_path)

    for cam_name, cam_token in sample['data'].items():
        if 'CAM' in cam_name:
            cam_path = nusc.get_sample_data_path(cam_token)
            frame.add_slot(f'{cam_name.lower()}_path', cam_path)


def add_annotations_to_frame(frame, nusc, sample):
    """Add annotations to a frame."""
    for ann_token in sample['anns']:
        ann = nusc.get('sample_annotation', ann_token)
        frame.add_slot(f'annotation_{ann_token}', {
            'category': ann['category_name'],
            'location': Coordinate(*ann['translation']),
            'size': ann['size'],
            'rotation': ann['rotation'],
            'velocity': nusc.box_velocity(ann_token),
            'num_lidar_pts': ann['num_lidar_pts'],
            'num_radar_pts': ann['num_radar_pts']
        })