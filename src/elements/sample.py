# src/elements/sample.py
from typing import Dict, List
from .camera_data import CameraData  # Ensure this path matches your setup

class Sample:
    def __init__(self, token: str, timestamp: int, scene_token: str):
        self.token = token
        self.timestamp = timestamp
        self.scene_token = scene_token
        self.camera_data: Dict[str, CameraData] = {}
        self.annotations: List[str] = []  # Store annotation tokens

    def add_camera_data(self, camera_name: str, camera_data: CameraData):
        self.camera_data[camera_name] = camera_data

    def add_annotations(self, annotation_tokens: List[str]):
        self.annotations = annotation_tokens
