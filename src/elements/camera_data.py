# src/elements/camera_data.py
import numpy as np

class CameraData:
    def __init__(self, intrinsics: np.ndarray, extrinsics: np.ndarray, image_path: str):
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.image_path = image_path