from typing import List

class SceneBox:
    def __init__(self, aabb: List[List[float]], near: float, far: float, radius: float, collider_type: str):
        self.aabb = aabb
        self.near = near
        self.far = far
        self.radius = radius
        self.collider_type = collider_type