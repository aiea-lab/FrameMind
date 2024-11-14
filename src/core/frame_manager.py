from typing import Any, Dict, List, Optional
from core.frame import Frame
from core.status import Status
from core.coordinate import Coordinate
from datetime import datetime
import numpy as np

class FrameNotFoundError(Exception):
    pass

class FrameManager:
    def __init__(self):
        self.frames: List[Frame] = []
        self.trajectories = {}

    def create_frame(self, name: str, timestamp: datetime, status: Status, coordinates: Coordinate, elements=None) -> Frame:
        if elements is None:
            elements = []  # or some other default value
        frame = Frame(name, elements, timestamp, status, coordinates)
        self.frames.append(frame)
        return frame

    def get_frame(self, name: str) -> Optional[Frame]:
        return next((frame for frame in self.frames if frame.name == name), None)

    def add_frame_data(self, frame_name: str, slot_name: str, value: Any):
        frame = self.get_frame(frame_name)
        if frame:
            frame.add_slot(slot_name, value)

    def get_frame_data(self, frame_name: str, slot_name: str) -> Any:
        frame = self.get_frame(frame_name)
        if frame:
            return frame.get_slot(slot_name)
        return None
    
    def create_frame_from_nuscenes(self, sample_annotation: Dict[str, Any]) -> Frame:
        """
        Create a frame from a nuScenes sample annotation.
        """
        frame = Frame(
            name=sample_annotation['token'],
            timestamp=datetime.fromtimestamp(sample_annotation['timestamp'] / 1e6),  # nuScenes uses microseconds
            status=Status.ACTIVE,  # You might want to determine this based on the annotation
            coordinates=Coordinate(
                latitude=sample_annotation['location'][0],
                longitude=sample_annotation['location'][1]
            )
        )
        frame.add_nuscenes_annotation(sample_annotation)
        self.frames.append(frame)
        return frame

    def add_nuscenes_annotation_to_frame(self, frame_name: str, annotation: Dict[str, Any]):
        frame = self.get_frame(frame_name)
        frame.add_nuscenes_annotation(annotation)

    def bulk_create_frames(self, frame_data: List[Dict[str, Any]]) -> List[Frame]:
        return [self.create_frame(**data) for data in frame_data]

    def search_frames(self, **kwargs) -> List[Frame]:
        return [frame for frame in self.frames if all(getattr(frame, k) == v for k, v in kwargs.items())]

    def delete_frame(self, name: str):
        frame = self.get_frame(name)
        self.frames.remove(frame)
        del self.frame_index[name]

    def update_frame(self, name: str, **kwargs):
        frame = self.get_frame(name)
        for key, value in kwargs.items():
            setattr(frame, key, value)

    def get_all_frames(self) -> List[Frame]:
        return self.frames.copy()
    
    def get_number_of_frames(self) -> int:
        """Returns the total number of frames managed by FrameManager."""
        return len(self.frames)
    
    def add_nuscenes_annotation_to_frame(self, frame_name: str, annotation: Dict[str, Any]):
        """Add nuScenes annotation data to an existing frame."""
        frame = self.get_frame(frame_name)
        if frame:
            frame.category = annotation.get('category', frame.category)
            frame.num_lidar_pts = annotation.get('num_lidar_pts', frame.num_lidar_pts)
            frame.num_radar_pts = annotation.get('num_radar_pts', frame.num_radar_pts)
            frame.instance_token = annotation.get('instance_token', frame.instance_token)
            frame.visibility_token = annotation.get('visibility_token', frame.visibility_token)
        else:
            raise FrameNotFoundError(f"Frame with name {frame_name} not found.")

    def add_neighbors(self, frame_name_1: str, frame_name_2: str):
        """Add two frames as neighbors to each other."""
        frame_1 = self.get_frame(frame_name_1)
        frame_2 = self.get_frame(frame_name_2)
        if frame_1 and frame_2:
            frame_1.add_neighbor(frame_2)
            frame_2.add_neighbor(frame_1)
        else:
            raise FrameNotFoundError(f"One or both frames not found: {frame_name_1}, {frame_name_2}")

    def simulate_communication(self):
        """Simulate frame communication."""
        for frame in self.frames:
            # Each frame shares its annotations with its neighbors
            frame.share_info_with_neighbors()

        # Each frame requests info from its neighbors
        for neighbor in frame.neighbor_frames:
            frame.request_info_from_frame(neighbor)

    def combine_similar_frames(self, distance_threshold: float, time_threshold: float):
        combined_frames = []
        visited = set()
        
        for i, frame1 in enumerate(self.frames):
            if i in visited:
                continue
            combined_frame = frame1
            for j, frame2 in enumerate(self.frames):
                if i != j and j not in visited:
                    if self.are_frames_similar(frame1, frame2, distance_threshold, time_threshold):
                        combined_frame = self.merge_frames(combined_frame, frame2)
                        visited.add(j)
            combined_frames.append(combined_frame)
            visited.add(i)
        
        self.frames = combined_frames

    def are_frames_similar(self, frame1: Frame, frame2: Frame, distance_threshold: float, time_threshold: float) -> bool:
       # Extract x, y, z coordinates from the Coordinate object
        coord1 = np.array([frame1.coordinates.x, frame1.coordinates.y, frame1.coordinates.z])
        coord2 = np.array([frame2.coordinates.x, frame2.coordinates.y, frame2.coordinates.z])
        
        # Calculate spatial distance between the two frames
        distance = np.linalg.norm(coord1 - coord2)
        
        # Calculate time difference in seconds
        time_diff = abs(frame1.timestamp - frame2.timestamp).total_seconds()
        
        # Return whether both spatial distance and time difference are within thresholds
        return distance <= distance_threshold and time_diff <= time_threshold

    def merge_frames(self, frame1: Frame, frame2: Frame) -> Frame:
        # Merge annotations and sensor data
        frame1.annotations.extend(frame2.annotations)
        frame1.slots.update(frame2.slots)  # Combine sensor slots (like lidar, camera data)
        return frame1

    def update_trajectories(self):
        """Update the trajectories for each object across frames."""
        for frame in self.frames:
            timestamp = frame.timestamp.timestamp()  # Convert datetime to a Unix timestamp
            for annotation in frame.annotations:
                object_id = annotation.get("id")  # Object ID in annotation
                location = annotation.get("location")  # Location in annotation

                if object_id and location:
                    # Ensure trajectory exists for this object
                    if object_id not in self.trajectories:
                        self.trajectories[object_id] = Trajectory(object_id=object_id)

                    # Update the trajectory with the current position and timestamp
                    self.trajectories[object_id].add_position(
                        (location.x, location.y, location.z), timestamp
                    )

    def get_object_trajectory(self, object_id: str):
        """Retrieve the trajectory for a specific object."""
        if object_id in self.trajectories:
            return self.trajectories[object_id].get_trajectory()
        else:
            raise ValueError(f"No trajectory found for object ID: {object_id}")

        def condense_frames_statistically(self, scene_id, scene_name, distance_threshold=5.0, time_threshold=2.0):
        """
        Condense frames by aggregating information within specified spatial and temporal thresholds and
        generate output in the specified format.
        
        Parameters:
            scene_id (str): Identifier for the scene.
            scene_name (str): Descriptive name of the scene.
            distance_threshold (float): Maximum distance between frames for them to be considered similar.
            time_threshold (float): Maximum time difference between frames for them to be considered similar.

        Returns:
            dict: Condensed scene data suitable for JSON serialization.
        """
        condensed_frames = []
        visited = set()
        
        for i, frame in enumerate(self.frames):
            if i in visited:
                continue
            
            # Initialize the condensed representation for a single frame
            condensed_frame = {
                "frame_id": frame.name,
                "average_timestamp": frame.timestamp.isoformat(),
                "average_position": [frame.coordinates.x, frame.coordinates.y, frame.coordinates.z],
                "annotations": frame.annotations[:],  # Start with annotations from the current frame
                "frame_count": 1
            }
            
            for j in range(i + 1, len(self.frames)):
                other_frame = self.frames[j]
                if j not in visited and self.are_frames_similar(frame, other_frame, distance_threshold, time_threshold):
                    # Update the condensed frame with data from other_frame
                    condensed_frame["frame_count"] += 1
                    condensed_frame["average_position"] = self._average_positions(
                        condensed_frame["average_position"], other_frame.coordinates, condensed_frame["frame_count"]
                    )
                    condensed_frame["annotations"] = self._merge_annotations(
                        condensed_frame["annotations"], other_frame.annotations
                    )
                    visited.add(j)
            
            condensed_frames.append(condensed_frame)
            visited.add(i)
        
        # Calculate scene-level metadata
        timestamps = [frame["average_timestamp"] for frame in condensed_frames]
        positions = [frame["average_position"] for frame in condensed_frames]
        
        timestamp_range = {
            "start": min(timestamps),
            "end": max(timestamps)
        }
        average_position = [
            sum(pos[0] for pos in positions) / len(positions),
            sum(pos[1] for pos in positions) / len(positions),
            sum(pos[2] for pos in positions) / len(positions),
        ]
        number_of_frames_combined = sum(frame["frame_count"] for frame in condensed_frames)
        
        # Create the condensed scene output
        scene_output = {
            "scene_id": scene_id,
            "scene_name": scene_name,
            "timestamp_range": timestamp_range,
            "average_position": average_position,
            "number_of_frames_combined": number_of_frames_combined,
            "condensed_frames": condensed_frames
        }
        
        return scene_output

    def _merge_annotations(self, existing_annotations, new_annotations):
        """
        Merge annotations from two frames, avoiding duplicates and aggregating relevant properties.
        
        Parameters:
            existing_annotations (list): List of annotations from the existing condensed frame.
            new_annotations (list): List of annotations from the frame being merged.

        Returns:
            list: Merged list of annotations.
        """
        merged_annotations = {  # Use a dictionary to avoid duplicates
            (ann["category"], tuple(ann["location"])): ann
            for ann in existing_annotations
        }
        
        for ann in new_annotations:
            key = (ann["category"], tuple(ann["location"]))
            if key in merged_annotations:
                # Merge properties like velocity, size, etc., if needed
                existing_ann = merged_annotations[key]
                existing_ann["velocity"] = self._average_velocity(
                    existing_ann.get("velocity", [0, 0, 0]),
                    ann.get("velocity", [0, 0, 0])
                )
                # Add other property merges as necessary
            else:
                merged_annotations[key] = ann
        
        return list(merged_annotations.values())

    def _average_positions(self, existing_position, new_position, count):
        """
        Compute the average position based on the new position and count.
        
        Parameters:
            existing_position (list): Current average position [x, y, z].
            new_position (Coordinate): New position to include in the average.
            count (int): Number of positions included in the average so far.

        Returns:
            list: Updated average position [x, y, z].
        """
        return [
            (existing_position[0] * (count - 1) + new_position.x) / count,
            (existing_position[1] * (count - 1) + new_position.y) / count,
            (existing_position[2] * (count - 1) + new_position.z) / count,
        ]


    def update_trajectories(self):
        """Update the trajectories for each object across frames."""
        for frame in self.frames:
            timestamp = frame.timestamp.timestamp()  # Convert datetime to a Unix timestamp
            for annotation in frame.annotations:
                object_id = annotation.get("id")  # Object ID in annotation
                location = annotation.get("location")  # Location in annotation

                # Debug: Check if object_id and location exist
                print(f"Processing frame {frame.name} - Object ID: {object_id}, Location: {location}")

                if object_id and location:
                    # Ensure trajectory exists for this object
                    if object_id not in self.trajectories:
                        self.trajectories[object_id] = Trajectory(object_id=object_id)

                    # Update the trajectory with the current position and timestamp
                    self.trajectories[object_id].add_position(
                        (location.x, location.y, location.z), timestamp
                    )

    def get_all_trajectories(self):
        """Retrieve all trajectories in a JSON-serializable format."""
        all_trajectories = []
        for object_id, trajectory in self.trajectories.items():
            trajectory_data = {
                "object_id": object_id,
                "positions": trajectory.positions,
                "timestamps": trajectory.timestamps,
            }
            all_trajectories.append(trajectory_data)
        return all_trajectories