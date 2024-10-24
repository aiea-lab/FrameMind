from condense_frames import FrameCondenser
from frame_manager import Frame, Annotation, Coordinate
import numpy as np

def validate_condensing():
    """
    Validates the frame condensing process with sample data.
    """
    # Sample frames data
    frame1 = Frame(
        coordinates=(608.85, 2011.05, 0.0),
        status='ACTIVE',
        annotations=[
            Annotation(
                category='vehicle.car',
                location=Coordinate(640.775, 1970.009, 1.143),
                size=[2.0, 4.32, 1.49],
                rotation=[0.9227, 0.0, 0.0, -0.3855],
                velocity=np.array([np.nan, np.nan, np.nan]),
                num_lidar_pts=0,
                num_radar_pts=1
            )
        ]
    )

    frame2 = Frame(
        coordinates=(610.2, 2012.1, 0.0),
        status='ACTIVE',
        annotations=[
            Annotation(
                category='vehicle.car',
                location=Coordinate(625.206, 1972.364, 0.772),
                size=[2.126, 4.592, 1.641],
                rotation=[0.3673, 0.0, 0.0, 0.9301],
                velocity=np.array([-6.88, 6.54, -0.36]),
                num_lidar_pts=8,
                num_radar_pts=2
            )
        ]
    )

    # Initialize the condenser
    condenser = FrameCondenser()

    # Condense frames
    condensed_scene = condenser.condense_frames([frame1, frame2])

    # Display the condensed scene
    print("Condensed Scene:", condensed_scene)

if __name__ == "__main__":
    validate_condensing()
