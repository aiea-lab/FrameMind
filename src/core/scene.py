# src/core/scene.py
from typing import List
from src.elements.sample import Sample  # Ensure the import path is correct

class Scene:
    def __init__(self, token: str, name: str):
        self.token = token
        self.name = name
        self.samples: List[Sample] = []

    def add_sample(self, sample: Sample):
        self.samples.append(sample)
