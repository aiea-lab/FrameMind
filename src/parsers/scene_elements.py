# src/parsers/scene_elements.py

import numpy as np
from typing import List

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
