from typing import List, Dict
import numpy as np
from datetime import datetime

class TrajectoryPoint:
    """Represents a single point in a trajectory, containing position and timestamp."""
    def __init__(self, timestamp: datetime, position: np.ndarray):
        self.object_id = object_id
        self.positions = []
        self.timestamps = []

class Trajectory:
    """Class that manages a trajectory, defined by a series of points over time."""
    def __init__(self, object_id: str):
        self.object_id = object_id
        self.points: List[TrajectoryPoint] = []

    def add_point(self, timestamp: datetime, position: List[float]):
        """Add a new point to the trajectory."""
        position_array = np.array(position)
        self.points.append(TrajectoryPoint(timestamp, position_array))

    def get_positions(self) -> np.ndarray:
        """Returns all positions in the trajectory as a NumPy array."""
        return np.array([point.position for point in self.points])

    def get_timestamps(self) -> List[datetime]:
        """Returns a list of all timestamps in the trajectory."""
        return [point.timestamp for point in self.points]

    def to_dict(self) -> Dict:
        """Convert the trajectory to a dictionary format."""
        return {
            "object_id": self.object_id,
            "trajectory": [
                {"timestamp": point.timestamp.isoformat(), "position": point.position.tolist()}
                for point in self.points
            ]
        }

    def add_position(self, position, timestamp):
        """Add a position and timestamp to the trajectory."""
        print(f"Adding position {position} at time {timestamp} for object {self.object_id}")  # Debug
        self.positions.append(position)
        self.timestamps.append(timestamp)

    def get_trajectory(self):
        """Return the trajectory data for output."""
        return {
            "object_id": self.object_id,
            "positions": self.positions,
            "timestamps": self.timestamps,
        }

class TrajectoryManager:
    """Manages trajectories for multiple objects in a scene."""
    def __init__(self):
        self.trajectories: Dict[str, Trajectory] = {}

    def add_point(self, object_id: str, timestamp: datetime, position: List[float]):
        """Add a point to an object's trajectory, creating a new trajectory if necessary."""
        if object_id not in self.trajectories:
            self.trajectories[object_id] = Trajectory(object_id)
        self.trajectories[object_id].add_point(timestamp, position)

    def get_trajectory(self, object_id: str) -> Trajectory:
        """Retrieve the trajectory for a specific object."""
        if object_id in self.trajectories:
            return self.trajectories[object_id]
        else:
            raise ValueError(f"No trajectory found for object ID: {object_id}")

    def get_all_trajectories(self) -> List[Trajectory]:
        """Returns all trajectories as a list."""
        return list(self.trajectories.values())

    def to_dict(self) -> List[Dict]:
        """Convert all trajectories to a dictionary format."""
        return [traj.to_dict() for traj in self.trajectories.values()]
