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

    def test_check_missing_data(self):
        # Test with no data added
        required_data_types = [Status.Camera, Status.SensorReading]
        missing_data = self.frame.check_missing_data(required_data_types)
        self.assertEqual(missing_data, [Status.Camera, Status.SensorReading])

        # Test with one data type added
        self.frame_manager.add_frame_data("Frame1", Status.Camera, {"image": "image_data"})
        missing_data = self.frame.check_missing_data(required_data_types)
        self.assertEqual(missing_data, [Status.SensorReading])

        # Test with all required data added
        self.frame_manager.add_frame_data("Frame1", Status.SensorReading, {"sensor": "LiDAR", "reading": [1, 2, 3]})
        missing_data = self.frame.check_missing_data(required_data_types)
        self.assertEqual(missing_data, [])
    
    def test_update_nonexistent_frame(self):
        self.frame_manager.update_frame_status("NonexistentFrame", Status.Completed)
        frame = self.frame_manager.get_frame("NonexistentFrame")
        self.assertIsNone(frame)

    def test_remove_frame_data(self):
        self.frame_manager.add_frame_data("Frame1", Status.Camera, {"image": "image_data"})
        frame = self.frame_manager.get_frame("Frame1")
        frame.data.pop(Status.Camera, None)
        camera_data = self.frame_manager.get_frame_data("Frame1", Status.Camera)
        self.assertEqual(camera_data, [])

    def test_edge_cases_coordinates(self):
        with self.assertRaises(ValueError):
            coord_invalid = Coordinate(latitude=100.0, longitude=190.0)  # Invalid coordinates
    def test_frame_manager_initialization_and_cleanup(self):
        new_frame_manager = FrameManager()
        self.assertEqual(len(new_frame_manager.frames), 0)
        new_frame_manager.create_frame(
            frame_name="TestFrame",
            timestamp=datetime.now(),
            status=Status.Active,
            coordinates=self.coord
        )
        self.assertEqual(len(new_frame_manager.frames), 1)
        new_frame_manager.frames.clear()
        self.assertEqual(len(new_frame_manager.frames), 0)

    def test_add_data_to_nonexistent_frame(self):
        self.frame_manager.add_frame_data("NonexistentFrame", Status.Camera, {"image": "image_data"})
        frame = self.frame_manager.get_frame("NonexistentFrame")
        self.assertIsNone(frame)

    def test_add_invalid_data_type(self):
        with self.assertRaises(TypeError):
            self.frame_manager.add_frame_data("Frame1", "InvalidStatus", {"image": "image_data"})  # Invalid data type

    def test_frame_with_different_timestamps(self):
        timestamp1 = datetime.now()
        timestamp2 = datetime.now()
        frame1 = self.frame_manager.create_frame("Frame2", timestamp1, Status.Active, self.coord)
        frame2 = self.frame_manager.create_frame("Frame3", timestamp2, Status.Active, self.coord)
        self.assertNotEqual(frame1.timestamp, frame2.timestamp)

    def test_update_frame_data(self):
        self.frame_manager.add_frame_data("Frame1", Status.Camera, {"image": "image_data"})
        self.frame_manager.add_frame_data("Frame1", Status.Camera, {"image": "updated_image_data"})
        camera_data = self.frame_manager.get_frame_data("Frame1", Status.Camera)
        self.assertEqual(len(camera_data), 2)
        self.assertEqual(camera_data[1], {"image": "updated_image_data"})

    def test_remove_frame(self):
        frame_to_remove = self.frame_manager.create_frame("FrameToRemove", datetime.now(), Status.Active, self.coord)
        self.frame_manager.frames.remove(frame_to_remove)
        frame = self.frame_manager.get_frame("FrameToRemove")
        self.assertIsNone(frame)

    def test_retrieve_frames_by_status(self):
        self.frame_manager.create_frame("Frame2", datetime.now(), Status.Completed, self.coord)
        self.frame_manager.create_frame("Frame3", datetime.now(), Status.Completed, self.coord)
        completed_frames = [frame for frame in self.frame_manager.get_all_frames() if frame.status == Status.Completed]
        self.assertEqual(len(completed_frames), 2)


if __name__ == "__main__":
    unittest.main()