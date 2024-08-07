from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from datetime import datetime
import os

from frame_structure import Status, Coordinate, FrameManager

def initialize_nuscenes(version: str, dataroot: str):
    return NuScenes(version=version, dataroot=dataroot, verbose=True)

def extract_timestamp(sample):
    return datetime.utcfromtimestamp(sample['timestamp'] / 1e6)

def create_frame_from_sample(nusc, sample, frame_manager):
    frame_name = sample['token']
    timestamp = extract_timestamp(sample)
    status = Status.Active  # Example status
    
    # Assuming these coordinates are placeholders; real coordinates need to be sourced correctly
    coordinates = Coordinate(
        latitude=0.0,  # Replace with correct data
        longitude=0.0  # Replace with correct data
    )

    frame = frame_manager.create_frame(
        frame_name=frame_name,
        timestamp=timestamp,
        status=status,
        coordinates=coordinates
    )

    for sensor_name in ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']:
        sensor_token = sample['data'][sensor_name]
        sensor_data = nusc.get('sample_data', sensor_token)
        frame_manager.add_frame_data(frame_name, Status.Camera, sensor_data)

    lidar_token = sample['data']['LIDAR_TOP']
    lidar_data = nusc.get('sample_data', lidar_token)
    lidar_pointcloud = LidarPointCloud.from_file(os.path.join(nusc.dataroot, lidar_data['filename']))
    frame_manager.add_frame_data(frame_name, Status.SensorReading, lidar_pointcloud.points)

    for radar_name in ['RADAR_FRONT', 'RADAR_BACK', 'RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT', 'RADAR_BACK_LEFT', 'RADAR_BACK_RIGHT']:
        radar_token = sample['data'][radar_name]
        radar_data = nusc.get('sample_data', radar_token)
        radar_pointcloud = RadarPointCloud.from_file(os.path.join(nusc.dataroot, radar_data['filename']))
        frame_manager.add_frame_data(frame_name, Status.SensorReading, radar_pointcloud.points)

    return frame