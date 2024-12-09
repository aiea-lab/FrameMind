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
        self.time_window = getattr(config, 'time_window', 0.2)

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

    def merge_object_frames(self, base_frame, frames_to_merge):
        """
        Merge multiple object frames into one, with error handling for empty frames.
        """
        try:
            object_id = (
                base_frame.get('raw_object_id') or  # Check direct object ID first
                base_frame.get('object_id') or      # Alternative field name
                (base_frame.get('motion_metadata', [{}])[0].get('raw_object_id')  # Check in metadata
                if base_frame.get('motion_metadata') else None) or
                'unknown'  # Fallback value
            )
            # Check if base_frame has valid motion metadata
            if not base_frame.get('motion_metadata') or len(base_frame['motion_metadata']) == 0:
                print(f"Warning: Base frame has no motion metadata")
                return None

            merged_frame = {
                "raw_object_id": object_id,
                "raw_category": base_frame.get('raw_category', 'unknown'),
                "raw_static_size": base_frame.get('raw_static_size', [0, 0, 0]),
                "motion_metadata": []
            }

            # Collect all valid motion metadata
            all_metadata = []
            for frame in [base_frame] + frames_to_merge:
                if frame.get('motion_metadata'):
                    all_metadata.extend(frame['motion_metadata'])

            # Sort by timestamp if available
            all_metadata.sort(key=lambda x: x.get('raw_timestamp', ''))
            
            # Remove duplicates based on timestamp
            seen_timestamps = set()
            unique_metadata = []
            for metadata in all_metadata:
                timestamp = metadata.get('raw_timestamp')
                if timestamp and timestamp not in seen_timestamps:
                    seen_timestamps.add(timestamp)
                    unique_metadata.append(metadata)

            merged_frame['motion_metadata'] = unique_metadata
            return merged_frame

        except Exception as e:
            print(f"Error merging object frames: {str(e)}")
            return None
        
    def condense_frames(self, frames):
        """
        Condense object frames with proper error handling.
        """
        try:
            if not frames:
                print("Warning: No frames to condense")
                return []

            condensed_frames = []
            current_group = []
            
            for frame in frames:
                # Skip frames with no motion metadata
                if not frame.get('motion_metadata'):
                    continue
                    
                if not current_group:
                    current_group = [frame]
                else:
                    # Check if frame should be merged with current group
                    if self._should_merge_frame(current_group[0], frame):
                        current_group.append(frame)
                    else:
                        # Merge current group and start new one
                        merged = self.merge_object_frames(current_group[0], current_group[1:])
                        if merged:
                            condensed_frames.append(merged)
                        current_group = [frame]

            # Handle last group
            if current_group:
                merged = self.merge_object_frames(current_group[0], current_group[1:])
                if merged:
                    condensed_frames.append(merged)

            return condensed_frames

        except Exception as e:
            print(f"Error in frame condensation: {str(e)}")
            return []

    def _should_merge_frame(self, frame1, frame2):
        """
        Determine if two frames should be merged.
        """
        try:
            # Get metadata from frames
            metadata1 = frame1.get('motion_metadata', [])
            metadata2 = frame2.get('motion_metadata', [])
            
            # Skip if either frame has no metadata
            if not metadata1 or not metadata2:
                return False
                
            # Compare timestamps
            time1 = metadata1[0].get('raw_timestamp', '')
            time2 = metadata2[0].get('raw_timestamp', '')
            
            if not time1 or not time2:
                return False
                
            # Convert timestamps to datetime objects
            dt1 = datetime.fromisoformat(time1)
            dt2 = datetime.fromisoformat(time2)
            
            # Check if frames are within time window
            time_diff = abs((dt2 - dt1).total_seconds())
            return time_diff <= self.config.time_window

        except Exception as e:
            print(f"Error checking frame merge compatibility: {str(e)}")
            return False