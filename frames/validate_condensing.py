import numpy as np
from frame_manager import Frame, Coordinate, Annotation, Status
from collections import Counter

class FrameCondenser:
    """
    Class to handle the process of condensing multiple frames into a single scene.
    """

    def __init__(self):
        pass

    def condense_frames(self, frames: list) -> Frame:
        """
        Condense a list of frames into a single frame by averaging coordinates.
        """
        if not frames:
            return None

        # Initialize summed coordinates
        summed_x = 0.0
        summed_y = 0.0
        summed_z = 0.0
        all_annotations = []
        status_list = []

        # Sum up all the coordinates from each frame
        for frame in frames:
            summed_x += frame.coordinates.x
            summed_y += frame.coordinates.y
            summed_z += frame.coordinates.z
            all_annotations.extend(frame.annotations)
            status_list.append(frame.status)

        # Calculate the average coordinates
        num_frames = len(frames)
        avg_coordinates = Coordinate(
            x=summed_x / num_frames,
            y=summed_y / num_frames,
            z=summed_z / num_frames
        )

        # Determine the most common status
        most_common_status = Counter(status_list).most_common(1)[0][0]

        # Create a new condensed frame with averaged coordinates
        condensed_frame = Frame(
            name='condensed_frame',
            timestamp=frames[0].timestamp,  # Using timestamp from the first frame
            status=most_common_status,
            coordinates=avg_coordinates,
            annotations=all_annotations
        )

        return condensed_frame

    def _merge_annotations(self, frames):
        """
        Merges annotations from multiple frames into a single list of annotations.
        
        Args:
            frames (list): List of frame objects with annotations.
            
        Returns:
            list: Merged annotations for the condensed scene.
        """
        merged_annotations = []

        # Start with annotations from the first frame
        base_annotations = frames[0].annotations

        for base_obj in base_annotations:
            matching_objs = [base_obj]
            
           
            for frame in frames[1:]:
                for other_obj in frame.annotations:
                    if other_obj.category == base_obj.category:
                        matching_objs.append(other_obj)
                        break
            
            # Calculate average attributes for the merged object
            avg_location = np.mean(
                [[obj.location.x, obj.location.y, obj.location.z] for obj in matching_objs], axis=0
            )
            avg_size = np.mean([obj.size for obj in matching_objs], axis=0)
            avg_rotation = np.mean([obj.rotation for obj in matching_objs], axis=0)

            # Handle velocity, replacing NaN with 0
            avg_velocity = np.nan_to_num(
                [np.nanmean([obj.velocity[i] for obj in matching_objs]) for i in range(3)]
            )

            # Sum lidar and radar points
            total_lidar_pts = sum(obj.num_lidar_pts for obj in matching_objs)
            total_radar_pts = sum(obj.num_radar_pts for obj in matching_objs)

            # Create the merged object annotation

            
            merged_obj = {
                'category': base_obj.category,
                'location': avg_location,
                'size': avg_size,
                'rotation': avg_rotation,
                'velocity': avg_velocity,
                'num_lidar_pts': total_lidar_pts,
                'num_radar_pts': total_radar_pts
            }
            merged_annotations.append(merged_obj)

        return merged_annotations
