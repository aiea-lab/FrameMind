from typing import List, Tuple
import numpy as np

class Trajectory:
    def __init__(self, object_id: str):
        self.object_id = object_id
        self.positions: List[Tuple[float, float, float]] = []  # (x, y, z) positions
        self.timestamps: List[float] = []  # Timestamps for each position

    def add_position(self, position: Tuple[float, float, float], timestamp: float):
        """Add a new position for the object at a given time."""
        self.positions.append(position)
        self.timestamps.append(timestamp)

    def calculate_distances(self) -> List[float]:
        """Calculate distances between consecutive positions."""
        distances = []
        for i in range(1, len(self.positions)):
            dist = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[i - 1]))
            distances.append(dist)
        return distances

    def calculate_speeds(self) -> List[float]:
        """Calculate speed between consecutive positions."""
        speeds = []
        for i in range(1, len(self.positions)):
            distance = np.linalg.norm(np.array(self.positions[i]) - np.array(self.positions[i - 1]))
            time_delta = self.timestamps[i] - self.timestamps[i - 1]
            speed = distance / time_delta if time_delta > 0 else 0
            speeds.append(speed)
        return speeds

    def get_trajectory(self) -> List[Tuple[float, float, float]]:
        """Get the complete trajectory as a list of positions."""
        return self.positions
