class Coordinate:
    def __init__(self, latitude: float, longitude: float):
        self.latitude = latitude
        self.longitude = longitude

    def __repr__(self):
        return f"Coordinate(latitude={self.latitude}, longitude={self.longitude})"