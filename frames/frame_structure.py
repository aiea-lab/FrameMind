from typing import List, Union, Tuple
from enum import Enum
from datetime import datetime

class Status(Enum):
    Active = 'Active'
    Idle = 'Idle'
    Operational = 'Operational'
    Malfunctioning = 'Malfunctioning'
    InProgress = 'InProgress'
    Completed = 'Completed'
    Aborted = 'Aborted'
    Connected = 'Connected'
    Disconnected = 'Disconnected'
    Planned = 'Planned'
    Minor = 'Minor'
    Major = 'Major'
    Critical = 'Critical'
    Collision = 'Collision'
    ObstacleDetected = 'ObstacleDetected'
    LaneDeparture = 'LaneDeparture'
    Object = 'Object'
    Event = 'Event'
    Camera = 'Camera'
    LiDAR = 'LiDAR'
    Radar = 'Radar'
    Image = 'Image'
    PointCloud = 'PointCloud'
    RadarSignal = 'RadarSignal'
    Info = 'Info'
    Warning = 'Warning'
    LogError = 'LogError'
    SensorReading = 'SensorReading'
    LogMessage = 'LogMessage'
    ExternalReport = 'ExternalReport'

class Coordinate:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude
        
    def __repr__(self):
        return f"Coordinate(latitude={self.latitude}, longitude={self.longitude})"


class Frame:
    def __init__(self, frame_name: str, timestamp: datetime, status: Status, coordinates: Coordinate):
        self.frame_name = frame_name
        self.timestamp = timestamp
        self.status = status
        self.coordinates = coordinates
        self.data = {}

    def add_data(self, data_type: Status, data: Union[str, List, dict]):
        if data_type not in self.data:
            self.data[data_type] = []
        self.data[data_type].append(data)

    def get_data(self, data_type: Status):
        return self.data.get(data_type, [])

    def __repr__(self):
        return f"Frame(name={self.frame_name}, timestamp={self.timestamp}, status={self.status}, coordinates=({self.coordinates.latitude}, {self.coordinates.longitude}))"

class FrameManager:
    def __init__(self):
        self.frames = []

    def create_frame(self, frame_name: str, timestamp: datetime, status: Status, coordinates: Coordinate):
        frame = Frame(frame_name, timestamp, status, coordinates)
        self.frames.append(frame)
        return frame

    def get_frame(self, frame_name: str) -> Union[Frame, None]:
        for frame in self.frames:
            if frame.frame_name == frame_name:
                return frame
        return None

    def get_all_frames(self) -> List[Frame]:
        return self.frames

    def update_frame_status(self, frame_name: str, new_status: Status):
        frame = self.get_frame(frame_name)
        if frame:
            frame.status = new_status

    def add_frame_data(self, frame_name: str, data_type: Status, data: Union[str, List, dict]):
        frame = self.get_frame(frame_name)
        if frame:
            frame.add_data(data_type, data)

    def get_frame_data(self, frame_name: str, data_type: Status):
        frame = self.get_frame(frame_name)
        if frame:
            return frame.get_data(data_type)
        return []
    
    
if __name__ == "__main__":
    # Create a frame manager
    frame_manager = FrameManager()

    # Create a coordinate
    coord = Coordinate(latitude=37.7749, longitude=-122.4194)

    # Create a frame
    frame = frame_manager.create_frame(
        frame_name="Frame1",
        timestamp=datetime.now(),
        status=Status.Active,
        coordinates=coord
    )

    # Add some data to the frame
    frame_manager.add_frame_data("Frame1", Status.Camera, {"image": "image_data"})
    frame_manager.add_frame_data("Frame1", Status.SensorReading, {"sensor": "LiDAR", "reading": [1, 2, 3]})

    # Retrieve and print frame data
    camera_data = frame_manager.get_frame_data("Frame1", Status.Camera)
    sensor_data = frame_manager.get_frame_data("Frame1", Status.SensorReading)
    print(f"Camera Data: {camera_data}")
    print(f"Sensor Data: {sensor_data}")

    # Print all frames
    print(f"All Frames: {frame_manager.get_all_frames()}")

    