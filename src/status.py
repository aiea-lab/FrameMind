from enum import Enum

class Status(Enum):
    #Basic statuses
    ACTIVE = "Active"
    IDLE = "Idle"
    ERROR = "Error"
    
    # nuScenes-specific statuses
    DETECTED = "Detected"  # Object has been detected by sensors
    TRACKED = "Tracked"  # Object is being tracked across frames
    OCCLUDED = "Occluded"  # Object is partially or fully occluded
    OUT_OF_RANGE = "Out of Range"  # Object is outside the sensor range
    
    # Sensor-related statuses
    LIDAR_DETECTED = "LiDAR Detected"  # Object detected by LiDAR
    CAMERA_DETECTED = "Camera Detected"  # Object detected by camera
    RADAR_DETECTED = "Radar Detected"  # Object detected by radar
    SENSOR_FUSION = "Sensor Fusion"  # Object detected by multiple sensors
    
    # Motion-related statuses
    MOVING = "Moving"  # Object is in motion
    STATIONARY = "Stationary"  # Object is not moving
    ACCELERATING = "Accelerating"  # Object is increasing speed
    DECELERATING = "Decelerating"  # Object is decreasing speed
    
    # Interaction-related statuses
    INTERACTING = "Interacting"  # Object is interacting with other objects
    CROSSING = "Crossing"  # Object is crossing the path of the ego vehicle
    MERGING = "Merging"  # Object is merging into traffic
    
    # Prediction-related statuses
    PREDICTED = "Predicted"  # Future position/trajectory has been predicted
    UNCERTAIN = "Uncertain"  # High uncertainty in object's future state
    
    # Annotation-related statuses
    ANNOTATED = "Annotated"  # Object has been manually annotated
    AUTO_LABELED = "Auto Labeled"  # Object has been automatically labeled
    
    # Quality-related statuses
    HIGH_CONFIDENCE = "High Confidence"  # High confidence in detection/tracking
    LOW_CONFIDENCE = "Low Confidence"  # Low confidence in detection/tracking
    
    # Scene-specific statuses
    IN_FRAME = "In Frame"  # Object is within the current frame
    OUT_OF_FRAME = "Out of Frame"  # Object has left the current frame
    
    # Temporal statuses
    KEYFRAME = "Keyframe"  # Object is in a keyframe (2Hz in nuScenes)
    INTERPOLATED = "Interpolated"  # Object's state is interpolated between keyframes