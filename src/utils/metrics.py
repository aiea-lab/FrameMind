from typing import List, Dict
import numpy as np

def calculate_frame_distance(frame1: Dict, frame2: Dict) -> float:
    """Calculate Euclidean distance between frame positions."""
    pos1 = np.array(frame1['raw_location'])
    pos2 = np.array(frame2['raw_location'])
    return np.linalg.norm(pos1 - pos2)

def calculate_velocity_change(frame1: Dict, frame2: Dict) -> float:
    """Calculate change in velocity between frames."""
    vel1 = np.array(frame1['raw_velocity'])
    vel2 = np.array(frame2['raw_velocity'])
    return np.linalg.norm(vel2 - vel1)

def calculate_frame_quality(frame: Dict) -> float:
    """Calculate overall frame quality score."""
    visibility_score = frame['derived_visibility']
    lidar_score = min(1.0, frame['raw_num_lidar_points'] / 50)
    radar_score = min(1.0, frame.get('raw_num_radar_points', 0) / 10)
    
    return (visibility_score + lidar_score + radar_score) / 3