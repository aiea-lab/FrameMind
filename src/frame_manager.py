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
    
    def add_nuscenes_annotation_to_frame(self, frame_name: str, annotation: Dict[str, Any]):
        """Add nuScenes annotation data to an existing frame."""
        frame = self.get_frame(frame_name)
        if frame:
            frame.category = annotation.get('category', frame.category)
            frame.num_lidar_pts = annotation.get('num_lidar_pts', frame.num_lidar_pts)
            frame.num_radar_pts = annotation.get('num_radar_pts', frame.num_radar_pts)
            frame.instance_token = annotation.get('instance_token', frame.instance_token)
            frame.visibility_token = annotation.get('visibility_token', frame.visibility_token)
        else:
            raise FrameNotFoundError(f"Frame with name {frame_name} not found.")

    def add_neighbors(self, frame_name_1: str, frame_name_2: str):
        """Add two frames as neighbors to each other."""
        frame_1 = self.get_frame(frame_name_1)
        frame_2 = self.get_frame(frame_name_2)
        if frame_1 and frame_2:
            frame_1.add_neighbor(frame_2)
            frame_2.add_neighbor(frame_1)
        else:
            raise FrameNotFoundError(f"One or both frames not found: {frame_name_1}, {frame_name_2}")

    def simulate_communication(self):
        """Simulate frame communication."""
        for frame in self.frames:
            # Each frame shares its annotations with its neighbors
            frame.share_info_with_neighbors()

        # Each frame requests info from its neighbors
        for neighbor in frame.neighbor_frames:
            frame.request_info_from_frame(neighbor)