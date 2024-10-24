import numpy as np

class FrameCondenser:
    """
    Class to handle the process of condensing multiple frames into a single scene.
    """
    
    def __init__(self):
        pass

    def condense_frames(self, frames):
        """
        Condenses a list of frames into a single scene.
        
        Args:
            frames (list): List of frame objects with coordinates, status, and annotations.
            
        Returns:
            dict: A dictionary representing the condensed scene with aggregated data.
        """
        if not frames:
            raise ValueError("No frames provided for condensing.")
        
        # Initialize scene attributes
        scene_coords = [0.0, 0.0, 0.0]
        scene_status = None
        merged_annotations = []

        # Calculate average coordinates and status
        for frame in frames:
            # Sum frame coordinates
            scene_coords[0] += frame.coordinates[0]
            scene_coords[1] += frame.coordinates[1]
            scene_coords[2] += frame.coordinates[2]

            # Determine the overall status
            if scene_status is None:
                scene_status = frame.status
            elif scene_status != frame.status:
                scene_status = 'MIXED'
        
        # Average the coordinates
        num_frames = len(frames)
        scene_coords = [coord / num_frames for coord in scene_coords]

        # Merge object annotations
        if frames[0].annotations:
            merged_annotations = self._merge_annotations(frames)

        # Construct the final condensed scene
        condensed_scene = {
            'coordinates': scene_coords,
            'status': scene_status,
            'annotations': merged_annotations
        }

        return condensed_scene

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
