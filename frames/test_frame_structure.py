import unittest
from datetime import datetime
from frame_structure import Status, Coordinate, Frame, FrameManager

class TestFrameStructure(unittest.TestCase):

    def setUp(self):
        self.frame_manager = FrameManager()
        self.coord = Coordinate(latitude=37.7749, longitude=-122.4194)
        self.frame = self.frame_manager.create_frame(
            frame_name="Frame1",
            timestamp=datetime.now(),
            status=Status.Active,
            coordinates=self.coord
        )

    def test_create_frame(self):
        frame = self.frame_manager.get_frame("Frame1")
        self.assertIsNotNone(frame)
        self.assertEqual(frame.frame_name, "Frame1")
        self.assertEqual(frame.status, Status.Active)
        self.assertEqual(frame.coordinates.latitude, 37.7749)
        self.assertEqual(frame.coordinates.longitude, -122.4194)

    def test_add_frame_data(self):
        self.frame_manager.add_frame_data("Frame1", Status.Camera, {"image": "image_data"})
        self.frame_manager.add_frame_data("Frame1", Status.SensorReading, {"sensor": "LiDAR", "reading": [1, 2, 3]})
        camera_data = self.frame_manager.get_frame_data("Frame1", Status.Camera)
        sensor_data = self.frame_manager.get_frame_data("Frame1", Status.SensorReading)
        self.assertEqual(len(camera_data), 1)
        self.assertEqual(camera_data[0], {"image": "image_data"})
        self.assertEqual(len(sensor_data), 1)
        self.assertEqual(sensor_data[0], {"sensor": "LiDAR", "reading": [1, 2, 3]})

    def test_update_frame_status(self):
        self.frame_manager.update_frame_status("Frame1", Status.Completed)
        frame = self.frame_manager.get_frame("Frame1")
        self.assertEqual(frame.status, Status.Completed)

    def test_get_nonexistent_frame(self):
        frame = self.frame_manager.get_frame("NonexistentFrame")
        self.assertIsNone(frame)

    def test_get_frame_data_empty(self):
        data = self.frame_manager.get_frame_data("Frame1", Status.Warning)
        self.assertEqual(data, [])

    def test_get_all_frames(self):
        frames = self.frame_manager.get_all_frames()
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0].frame_name, "Frame1")

if __name__ == "__main__":
    unittest.main()