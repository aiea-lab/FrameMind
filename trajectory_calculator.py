import numpy as np

class Coordinate:
    def __init__(self, x, y, z):
        """
        Initialize a Coordinate object.
        :param x: X-coordinate of the object
        :param y: Y-coordinate of the object
        :param z: Z-coordinate of the object
        """
        self.x = x
        self.y = y
        self.z = z

    def __repr__(self):
        return f"Coordinate(x={self.x}, y={self.y}, z={self.z})"

class Annotation:
    def __init__(self, category, location, size, rotation, velocity, num_lidar_pts, num_radar_pts):
        """
        Initialize an Annotation object.
        :param category: Category of the object (e.g., 'vehicle.car')
        :param location: Initial position of the object (Coordinate object)
        :param size: Size of the object [length, width, height]
        :param rotation: Rotation of the object [qx, qy, qz, qw]
        :param velocity: Velocity vector [vx, vy, vz]
        :param num_lidar_pts: Number of LiDAR points detected
        :param num_radar_pts: Number of radar points detected
        """
        self.category = category
        self.location = location
        self.size = size
        self.rotation = rotation
        self.velocity = velocity
        self.num_lidar_pts = num_lidar_pts
        self.num_radar_pts = num_radar_pts

    def __repr__(self):
        return (f"Annotation(category={self.category}, location={self.location}, "
                f"size={self.size}, rotation={self.rotation}, velocity={self.velocity}, "
                f"num_lidar_pts={self.num_lidar_pts}, num_radar_pts={self.num_radar_pts})")

class TrajectoryCalculator:
    def __init__(self, annotations, time_interval=1.0, num_steps=10):
        """
        Initialize the TrajectoryCalculator with annotations.
        :param annotations: List of Annotation objects
        :param time_interval: Time interval between frames (default is 1 second)
        :param num_steps: Number of time intervals for trajectory computation (default is 10)
        """
        self.annotations = annotations
        self.time_interval = time_interval
        self.num_steps = num_steps

    def compute_trajectory(self):
        """
        Computes the trajectory for each annotation based on velocity and initial location.
        :return: List of dictionaries containing the trajectory data
        """
        trajectories = []

        for annotation in self.annotations:
            location = annotation.location
            velocity = annotation.velocity

            # List to store trajectory points
            trajectory_points = [location]

            # Check if velocity contains NaN values
            if np.isnan(velocity).any():
                # If velocity contains NaN, assume stationary trajectory
                for _ in range(1, self.num_steps + 1):
                    trajectory_points.append(location)
            else:
                # Compute the trajectory based on velocity
                for t in range(1, self.num_steps + 1):
                    new_x = location.x + velocity[0] * self.time_interval * t
                    new_y = location.y + velocity[1] * self.time_interval * t
                    new_z = location.z + velocity[2] * self.time_interval * t
                    trajectory_points.append(Coordinate(new_x, new_y, new_z))

            # Append the computed trajectory for the current annotation
            trajectories.append({
                'category': annotation.category,
                'trajectory': trajectory_points
            })

        return trajectories

    def display_trajectory(self, trajectories):
        """
        Display the computed trajectory points for each annotation.
        :param trajectories: List of dictionaries containing the trajectory data
        """
        for trajectory in trajectories:
            print(f"Category: {trajectory['category']}")
            for point in trajectory['trajectory']:
                print(f"  {point}")

# Example usage of the TrajectoryCalculator
if __name__ == "__main__":
    # Define sample annotations
    annotations = [
        Annotation(
            category="vehicle.car",
            location=Coordinate(0.0, 0.0, 0.0),
            size=[2.0, 4.32, 1.49],
            rotation=[0.9227, 0.0, 0.0, -0.3855],
            velocity=np.array([np.nan, np.nan, np.nan]),
            num_lidar_pts=0,
            num_radar_pts=1
        ),
        Annotation(
            category="vehicle.car",
            location=Coordinate(1.0, 2.0, 0.0),
            size=[2.126, 4.592, 1.641],
            rotation=[0.3673, 0.0, 0.0, 0.9301],
            velocity=np.array([-6.88, 6.54, -0.36]),
            num_lidar_pts=8,
            num_radar_pts=2
        )
    ]

    # Initialize the TrajectoryCalculator with annotations
    trajectory_calculator = TrajectoryCalculator(annotations, time_interval=1.0, num_steps=10)

    # Compute trajectories
    trajectories = trajectory_calculator.compute_trajectory()

    # Display the trajectories
    trajectory_calculator.display_trajectory(trajectories)
