from typing import Any, Dict, List, Optional
from .frame import Frame
from .status import Status
from .coordinate import Coordinate
from datetime import datetime

class FrameManager:
    def __init__(self):
        self.frames: List[Frame] = []

    def create_frame(self, name: str, timestamp: datetime, status: Status, coordinates: Coordinate) -> Frame:
        frame = Frame(name, timestamp, status, coordinates)
        self.frames.append(frame)
        return frame

    def get_frame(self, name: str) -> Optional[Frame]:
        return next((frame for frame in self.frames if frame.name == name), None)

    def add_frame_data(self, frame_name: str, slot_name: str, value: Any):
        frame = self.get_frame(frame_name)
        if frame:
            frame.add_slot(slot_name, value)

    def get_frame_data(self, frame_name: str, slot_name: str) -> Any:
        frame = self.get_frame(frame_name)
        if frame:
            return frame.get_slot(slot_name)
        return None
    
    def create_frame_from_nuscenes(self, sample_annotation: Dict[str, Any]) -> Frame:
        """
        Create a frame from a nuScenes sample annotation.
        """
        frame = Frame(
            name=sample_annotation['token'],
            timestamp=datetime.fromtimestamp(sample_annotation['timestamp'] / 1e6),  # nuScenes uses microseconds
            status=Status.ACTIVE,  # You might want to determine this based on the annotation
            coordinates=Coordinate(
                latitude=sample_annotation['location'][0],
                longitude=sample_annotation['location'][1]
            )
        )
        frame.add_nuscenes_annotation(sample_annotation)
        self.frames.append(frame)
        return frame

    def add_nuscenes_annotation_to_frame(self, frame_name: str, annotation: Dict[str, Any]):
        """
        Add nuScenes annotation to an existing frame.
        """
        frame = self.get_frame(frame_name)
        if frame:
            frame.add_nuscenes_annotation(annotation)