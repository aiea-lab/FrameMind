import numpy as np
from typing import List

class Cameras:
    def __init__(self, poses: np.ndarray, intrinsics: np.ndarray, image_filenames: List[str]):
        self.poses = poses
        self.intrinsics = intrinsics
        self.image_filenames = image_filenames