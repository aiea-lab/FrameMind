from typing import Tuple, List
import math

class Coordinate:
    def __init__(self, x: float, y: float, z: float = 0.0):
        """
        Initialize a Coordinate object.
        
        :param x: x-coordinate in the local coordinate system (forward direction)
        :param y: y-coordinate in the local coordinate system (left direction)
        :param z: z-coordinate in the local coordinate system (up direction)
        """
        self.x = x
        self.y = y
        self.z = z
        self.latitude = y  # Assuming y represents latitude
        self.longitude = x  # Assuming x represents longitude
        self.altitude = z  # If z is provided

    @classmethod
    def from_nuscenes_location(cls, location: List[float]) -> 'Coordinate':
        """
        Create a Coordinate object from a nuScenes location list.
        
        :param location: List of [x, y, z] coordinates from nuScenes dataset
        :return: Coordinate object
        """
        return cls(location[0], location[1], location[2])

    def to_nuscenes_format(self) -> List[float]:
        """
        Convert the coordinate to nuScenes format.
        
        :return: List of [x, y, z] coordinates
        """
        return [self.x, self.y, self.z]

    def distance_to(self, other: 'Coordinate') -> float:
        """
        Calculate the Euclidean distance to another coordinate.
        
        :param other: Another Coordinate object
        :return: Distance in the same unit as the coordinates
        """
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2 + (self.z - other.z)**2)

    def to_2d(self) -> Tuple[float, float]:
        """
        Return the 2D projection of the coordinate.
        
        :return: Tuple of (x, y) coordinates
        """
        return (self.x, self.y)

    def __repr__(self):
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"

    def __eq__(self, other):
        if not isinstance(other, Coordinate):
            return NotImplemented
        return (self.x, self.y, self.z) == (other.x, other.y, other.z)

    def __hash__(self):
        return hash((self.x, self.y, self.z))
    
    def to_dict(self):
        return {"x": self.x, "y": self.y, "z": self.z}