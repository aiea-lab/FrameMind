from typing import Any, Dict, List, Optional
from frame import Frame
from status import Status
from coordinate import Coordinate
from datetime import datetime

class FrameNotFoundError(Exception):
    pass

class FrameManager:
    def __init__(self):
        self.frames: List[Frame] = []

    def create_frame(self, name: str, timestamp: datetime, status: Status, coordinates: Coordinate, elements=None) -> Frame:
        if elements is None:
            elements = []  # or some other default value
        frame = Frame(name, elements, timestamp, status, coordinates)
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
        frame = self.get_frame(frame_name)
        frame.add_nuscenes_annotation(annotation)

    def bulk_create_frames(self, frame_data: List[Dict[str, Any]]) -> List[Frame]:
        return [self.create_frame(**data) for data in frame_data]

    def search_frames(self, **kwargs) -> List[Frame]:
        return [frame for frame in self.frames if all(getattr(frame, k) == v for k, v in kwargs.items())]

    def delete_frame(self, name: str):
        frame = self.get_frame(name)
        self.frames.remove(frame)
        del self.frame_index[name]

    def update_frame(self, name: str, **kwargs):
        frame = self.get_frame(name)
        for key, value in kwargs.items():
            setattr(frame, key, value)

    def get_all_frames(self) -> List[Frame]:
        return self.frames.copy()