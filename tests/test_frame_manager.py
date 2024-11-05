import unittest
from datetime import datetime
from enum import Enum
from typing import List, Union

# Assuming the classes are imported from your module
from your_module import Status, Coordinate, Frame, FrameManager

class TestCoordinate(unittest.TestCase):
    def test_coordinate_initialization(self):
        coord = Coordinate(37.7749, -122.4194)
        self.assertEqual(coord.latitude, 37.7749)
        self.assertEqual(coord.longitude, -122.4194)

class TestFrame(unittest.TestCase):
    def setUp(self):
        self.coord = Coordinate(37.7749, -122.4194)
        self.frame = Frame(
            frame_name="Frame1",
            timestamp=datetime.now(),
            status=Status.Active,
            coordinates=self.coord
        )

    def test_frame_initialization(self):
        self.assertEqual(self.frame.frame_name, "Frame1")
        self.assertEqual(self.frame.status, Status.Active)
        self.assertEqual(self.frame.coordinates.latitude, 37.7749)
        self.assertEqual(self.frame.coordinates.longitude, -122.4194)

    def test_add_data(self):
        self.frame.add_data(Status.Camera, {"image": "image_data"})
        self.assertEqual(self.frame.get_data(Status.Camera), [{"image": "image_data"}])

    def test_get_data(self):
        self.frame.add_data(Status.SensorReading, {"sensor": "LiDAR", "reading": [1, 2, 3]})
        data = self.frame.get_data(Status.SensorReading)
        self.assertEqual(data, [{"sensor": "LiDAR", "reading": [1, 2, 3]}])

class TestFrameManager(unittest.TestCase):
    def setUp(self):
        self.frame_manager = FrameManager()
        self.coord = Coordinate(37.7749, -122.4194)
        self.frame = self.frame_manager.create_frame(
            frame_name="Frame1",
            timestamp=datetime.now(),
            status=Status.Active,
            coordinates=self.coord
        )

    def test_create_frame(self):
        self.assertEqual(len(self.frame_manager.get_all_frames()), 1)
        self.assertEqual(self.frame_manager.get_all_frames()[0].frame_name, "Frame1")

    def test_get_frame(self):
        frame = self.frame_manager.get_frame("Frame1")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_name, "Frame1")

    def test_update_frame_status(self):
        self.frame_manager.update_frame_status("Frame1", Status.Completed)
        frame = self.frame_manager.get_frame("Frame1")
        self.assertEqual(frame.status, Status.Completed)

    def test_add_frame_data(self):
        self.frame_manager.add_frame_data("Frame1", Status.Camera, {"image": "image_data"})
        data = self.frame_manager.get_frame_data("Frame1", Status.Camera)
        self.assertEqual(data, [{"image": "image_data"}])

    def test_get_frame_data(self):
        self.frame_manager.add_frame_data("Frame1", Status.SensorReading, {"sensor": "LiDAR", "reading": [1, 2, 3]})
        data = self.frame_manager.get_frame_data("Frame1", Status.SensorReading)
        self.assertEqual(data, [{"sensor": "LiDAR", "reading": [1, 2, 3]}])
