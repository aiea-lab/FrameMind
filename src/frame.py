from datetime import datetime
from typing import Any, Dict, List, Optional
from status import Status
from coordinate import Coordinate

class Frame:
    def __init__(self, name: str,elements, timestamp: datetime, status: Status, coordinates: Coordinate):
        self.name = name
        self.elements = elements
        self.related_frames = []
        self.timestamp = timestamp
        self.status = status
        self.coordinates = coordinates
        self.slots: Dict[str, Any] = {}
        self.children: List['Frame'] = []
        self.parent: Optional['Frame'] = None
        self.events: List[str] = []
        self.errors: List[str] = []

        # nuScenes-specific attributes
        self.category: str = ""
        self.attribute_tokens: List[str] = []
        self.instance_token: str = ""
        self.visibility_token: str = ""
        self.num_lidar_pts: int = 0
        self.num_radar_pts: int = 0
        self.bbox_2d: Optional[List[float]] = None
        self.bbox_3d: Optional[List[float]] = None

    def add_slot(self, slot_name: str, value: Any):
        self.slots[slot_name] = value

    def get_slot(self, slot_name: str) -> Any:
        return self.slots.get(slot_name)

    def add_child(self, child_frame: 'Frame'):
        self.children.append(child_frame)
        child_frame.parent = self

    def add_event(self, event: str):
        self.events.append(event)

    def add_error(self, error: str):
        self.errors.append(error)

    def get_all_slots(self) -> Dict[str, Any]:
        all_slots = self.slots.copy()
        for child in self.children:
            all_slots.update(child.get_all_slots())
        return all_slots
    
    def add_related_frame(self, frame):
        self.related_frames.append(frame)

    def find_frame_by_name(self, name: str) -> Optional['Frame']:
        if self.name == name:
            return self
        for child in self.children:
            found = child.find_frame_by_name(name)
            if found:
                return found
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'name': self.name,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'coordinates': {
            'x': self.coordinates.x,
            'y': self.coordinates.y,
            'z': self.coordinates.z if hasattr(self.coordinates, 'z') else None
        },
            'slots': self.slots,
            'events': self.events,
            'errors': self.errors,
            'category': self.category,
            'attribute_tokens': self.attribute_tokens,
            'instance_token': self.instance_token,
            'visibility_token': self.visibility_token,
            'num_lidar_pts': self.num_lidar_pts,
            'num_radar_pts': self.num_radar_pts,
            'bbox_2d': self.bbox_2d,
            'bbox_3d': self.bbox_3d,
            'children': [child.to_dict() for child in self.children]
        }

    def __repr__(self):
        return f"Frame(name={self.name}, timestamp={self.timestamp}, status={self.status}, coordinates={self.coordinates}, events={len(self.events)}, errors={len(self.errors)}, children={len(self.children)})"