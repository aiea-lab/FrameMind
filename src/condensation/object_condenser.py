import numpy as np
from typing import List, Dict, Optional
from dataclasses import dataclass
from datetime import datetime
import hashlib
import logging

logger = logging.getLogger(__name__)

@dataclass
class MotionData:
    """Data class for motion metadata with raw NuScenes format."""
    raw_timestamp: str
    raw_location: List[float]
    raw_rotation: Optional[List[float]] = None
    raw_category: Optional[str] = None
    raw_object_id: Optional[str] = None
    sensor_fusion_confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict) -> 'MotionData':
        """Create MotionData from NuScenes annotation format."""
        required_fields = {'raw_location'}
        if not all(field in data for field in required_fields):
            missing = required_fields - set(data.keys())
            raise ValueError(f"Missing required fields: {missing}")

        category = data.get('raw_category', '')
        confidence = cls.calculate_category_confidence(category)

        return cls(
            raw_timestamp=data.get('raw_timestamp', ''),
            raw_location=list(map(float, data['raw_location'])),
            raw_rotation=list(map(float, data['raw_rotation'])) if 'raw_rotation' in data else None,
            raw_category=category,
            raw_object_id=data.get('raw_object_id'),
            sensor_fusion_confidence=confidence
        )

    @staticmethod
    def calculate_category_confidence(category: str) -> float:
        """Calculate confidence based on object category."""
        category_weights = {
            'vehicle.car': 0.8,
            'vehicle.truck': 0.75,
            'vehicle.bus': 0.75,
            'vehicle.bicycle': 0.6,
            'vehicle.motorcycle': 0.6,
            'human.pedestrian': 0.7,
            'vehicle.construction': 0.7
        }
        return category_weights.get(category, 0.5)

class ObjectCondenser:
    """Condenser for NuScenes object frames."""

    def __init__(self, config):
        self.config = config

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Convert timestamp string to float seconds."""
        if not timestamp_str:
            logger.warning("Empty timestamp encountered.")
            return 0.0
        try:
            return datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f').timestamp()
        except ValueError as e:
            logger.warning(f"Invalid timestamp format: {timestamp_str}. Error: {e}")
            return 0.0

    def _generate_frame_id(self, source_ids: List[Optional[str]]) -> str:
        """Generate a unique frame ID based on source frame IDs."""
        source_ids = [sid for sid in source_ids if sid]  # Filter out None values
        if not source_ids:
            logger.error("No valid source IDs found to generate frame_id.")
            return "unknown_frame_id"
        id_string = "_".join(sorted(source_ids))
        return hashlib.md5(id_string.encode()).hexdigest()

    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        """Condense multiple object frames into merged frames."""
        logger.info(f"Starting condensation with {len(frames)} frames")

        if not frames:
            logger.warning("No frames provided for condensation.")
            return []

        processed_frames = []
        for i, frame in enumerate(frames):
            try:
                annotations = frame.get('raw_annotations', [])
                processed_frames.append({
                    'frame_id': frame.get('raw_sample_id'),
                    'timestamp': self._parse_timestamp(frame.get('raw_timestamp', '')),
                    'motion_metadata': [
                        MotionData.from_dict({
                            'raw_timestamp': frame.get('raw_timestamp'),
                            'raw_location': annotation.get('raw_location', [0, 0, 0]),
                            'raw_rotation': annotation.get('raw_rotation'),
                            'raw_category': annotation.get('raw_category'),
                            'raw_object_id': annotation.get('raw_object_id')
                        }).__dict__ for annotation in annotations
                    ]
                })
            except Exception as e:
                logger.warning(f"Error processing frame {i}: {e}")

        logger.info(f"Processed {len(processed_frames)} valid frames")

        condensed_frames = []
        current_group = []

        for frame in processed_frames:
            if not current_group:
                current_group.append(frame)
                continue

            if self._can_merge_frames(current_group[-1], frame):
                current_group.append(frame)
            else:
                condensed_frame = self._merge_object_frames(current_group)
                if condensed_frame:
                    condensed_frames.append(condensed_frame)
                current_group = [frame]

        if current_group:
            condensed_frame = self._merge_object_frames(current_group)
            if condensed_frame:
                condensed_frames.append(condensed_frame)

        logger.info(f"Condensed to {len(condensed_frames)} frames")
        return condensed_frames

    def _can_merge_frames(self, frame1: Dict, frame2: Dict) -> bool:
        """Check if frames can be merged based on time and position criteria."""
        time_diff = abs(frame1['timestamp'] - frame2['timestamp'])
        if time_diff > self.config.time_window:
            return False

        objects1 = {m['raw_object_id'] for m in frame1['motion_metadata']}
        objects2 = {m['raw_object_id'] for m in frame2['motion_metadata']}
        return bool(objects1 & objects2)

    def _merge_object_frames(self, frames: List[Dict]) -> Optional[Dict]:
        """Merge multiple object frames into a condensed format."""
        if not frames:
            return None
        try:
            # Use the first frame as a base
            base_frame = frames[0]
            merged = {
                "frame_id": None,  # Will be computed later
                "raw_object_id": base_frame['motion_metadata'][0]['raw_object_id'],
                "raw_category": base_frame['motion_metadata'][0]['raw_category'],
                "raw_static_size": base_frame.get('raw_static_size', []),
                "condensed_motion_metadata": {},
                "condensed_info": {}
            }

            # Group motion metadata by object ID
            object_metadata = []
            source_frame_ids = []

            for frame in frames:
                for motion in frame['motion_metadata']:
                    if motion['raw_object_id'] == merged["raw_object_id"]:
                        object_metadata.append(motion)
                        source_frame_ids.append(motion['raw_timestamp'])

            if not object_metadata:
                logger.warning("No valid motion metadata found for merging.")
                return None

            # Compute weighted averages and merged fields
            confidences = np.array([m['derived_sensor_fusion_confidence'] for m in object_metadata])
            confidences = np.maximum(confidences, 0.1)  # Avoid zero weights
            weights = confidences / np.sum(confidences)

            # Average key fields
            locations = np.array([m['raw_location'] for m in object_metadata])
            velocities = np.array([m.get('raw_velocity', [0, 0, 0]) for m in object_metadata])
            sizes = np.array([m.get('derived_dynamic_size', [0, 0, 0]) for m in object_metadata])
            visibilities = np.mean([m['derived_visibility'] for m in object_metadata])

            avg_location = np.average(locations, weights=weights, axis=0).tolist()
            avg_velocity = np.average(velocities, weights=weights, axis=0).tolist()
            avg_size = np.average(sizes, weights=weights, axis=0).tolist()

            # Determine occlusion status (most frequent value)
            occlusion_status = max(
                [m['derived_occlusion_status'] for m in object_metadata],
                key=[m['derived_occlusion_status'] for m in object_metadata].count
            )

            avg_confidence = float(np.mean(confidences))

            # Condensed motion metadata
            latest_timestamp = max(object_metadata, key=lambda x: self._parse_timestamp(x['raw_timestamp']))['raw_timestamp']
            merged["condensed_motion_metadata"] = {
                "raw_location": avg_location,
                "raw_velocity": avg_velocity,
                "derived_dynamic_size": avg_size,
                "derived_visibility": visibilities,
                "derived_occlusion_status": occlusion_status,
                "derived_sensor_fusion_confidence": avg_confidence,
                "raw_timestamp": latest_timestamp
            }

            # Condensed info
            time_stamps = [self._parse_timestamp(m['raw_timestamp']) for m in object_metadata]
            time_span = max(time_stamps) - min(time_stamps)
            position_variance = float(np.var(locations, axis=0).mean())

            merged["condensed_info"] = {
                "frame_count": len(frames),
                "time_span": time_span,
                "position_variance": position_variance,
                "average_confidence": avg_confidence,
                "source_frame_ids": source_frame_ids
            }

            # Generate deterministic frame_id
            merged["frame_id"] = self._generate_frame_id(source_frame_ids)

            return merged
        except Exception as e:
            logger.error(f"Error merging object frames: {e}", exc_info=True)
            return None
