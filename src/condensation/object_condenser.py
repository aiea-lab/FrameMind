import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from .base_condenser import BaseCondenser
from .config import CondensationConfig
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MotionData:
    """Data class for motion metadata with raw NuScenes format"""
    raw_timestamp: str  # Changed to str to match NuScenes format
    raw_location: List[float]
    raw_rotation: Optional[List[float]] = None
    raw_category: Optional[str] = None
    raw_object_id: Optional[str] = None
    sensor_fusion_confidence: float = 0.0

    @classmethod
    def from_dict(cls, data: Dict) -> 'MotionData':
        """Create MotionData from NuScenes annotation format"""
        required_fields = {'raw_location'}
        if not all(field in data for field in required_fields):
            missing = required_fields - set(data.keys())
            raise ValueError(f"Missing required fields: {missing}")
        
        # Calculate sensor fusion confidence based on object category
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
        """Calculate confidence based on object category"""
        base_confidence = 0.5  # Default confidence
        
        # Category-specific confidence adjustments
        category_weights = {
            'vehicle.car': 0.8,
            'vehicle.truck': 0.75,
            'vehicle.bus': 0.75,
            'vehicle.bicycle': 0.6,
            'vehicle.motorcycle': 0.6,
            'human.pedestrian': 0.7,
            'vehicle.construction': 0.7
        }
        
        return category_weights.get(category, base_confidence)

class ObjectCondenser(BaseCondenser):
    """Condenser for NuScenes object frames"""
    
    def __init__(self, config: CondensationConfig):
        super().__init__(config)
        self._frame_cache = {}
        
    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Convert NuScenes timestamp string to float seconds"""
        try:
            dt = datetime.strptime(timestamp_str, '%Y-%m-%dT%H:%M:%S.%f')
            return dt.timestamp()
        except (ValueError, TypeError):
            logger.warning(f"Invalid timestamp format: {timestamp_str}")
            return 0.0

    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        """Condense multiple object frames into merged frames"""
        logger.info(f"Starting condensation with {len(frames)} frames")
        
        if not frames:
            return []

        # Convert raw annotations to motion metadata format
        processed_frames = []
        for i, frame in enumerate(frames):
            try:
                if 'raw_annotations' not in frame:
                    logger.warning(f"Frame {i} missing raw_annotations")
                    continue
                
                processed_frame = {
                    'frame_id': frame.get('raw_sample_id'),
                    'timestamp': self._parse_timestamp(frame.get('raw_timestamp', '')),
                    'motion_metadata': []
                }
                
                # Convert each annotation to motion metadata
                for annotation in frame['raw_annotations']:
                    try:
                        motion_data = MotionData.from_dict({
                            'raw_timestamp': frame.get('raw_timestamp'),
                            'raw_location': annotation['raw_location'],
                            'raw_rotation': annotation.get('raw_rotation'),
                            'raw_category': annotation.get('raw_category'),
                            'raw_object_id': annotation.get('raw_object_id')
                        })
                        processed_frame['motion_metadata'].append(motion_data.__dict__)
                    except Exception as e:
                        logger.warning(f"Error processing annotation: {e}")
                        continue
                
                if processed_frame['motion_metadata']:
                    processed_frames.append(processed_frame)
                
            except Exception as e:
                logger.warning(f"Error processing frame {i}: {e}")
                continue

        logger.info(f"Processed {len(processed_frames)} valid frames")

        if not processed_frames:
            logger.warning("No valid frames after processing")
            return []

        # Group frames by time windows
        condensed_frames = []
        current_group = []
        
        for frame in processed_frames:
            try:
                if not current_group:
                    current_group.append(frame)
                    continue
                
                if self._can_merge_frames(current_group[-1], frame):
                    current_group.append(frame)
                else:
                    merged = self._merge_object_frames(current_group)
                    if merged:
                        condensed_frames.append(merged)
                    current_group = [frame]
                    
            except Exception as e:
                logger.warning(f"Error in frame grouping: {e}", exc_info=True)
                current_group = [frame]
        
        # Process final group
        if current_group:
            try:
                merged = self._merge_object_frames(current_group)
                if merged:
                    condensed_frames.append(merged)
            except Exception as e:
                logger.error(f"Error merging final frame group: {e}", exc_info=True)
        
        logger.info(f"Condensed to {len(condensed_frames)} frames")
        return condensed_frames

    def _can_merge_frames(self, frame1: Dict, frame2: Dict) -> bool:
        """Check if frames can be merged based on time and position criteria"""
        try:
            # Dynamically adjust time window for complex scenes
            time_diff = abs(frame1['timestamp'] - frame2['timestamp'])
            if time_diff > self.config.time_window:
                return False

            # Check object overlap
            objects1 = {m['raw_object_id'] for m in frame1['motion_metadata']}
            objects2 = {m['raw_object_id'] for m in frame2['motion_metadata']}

            if not objects1.intersection(objects2):
                return False

            # Compare positions of overlapping objects
            for obj_id in objects1.intersection(objects2):
                pos1 = self._get_object_position(frame1, obj_id)
                pos2 = self._get_object_position(frame2, obj_id)
                if pos1 is not None and pos2 is not None:
                    if np.linalg.norm(np.array(pos2) - np.array(pos1)) > self.config.max_position_gap:
                        return False

            return True
        except Exception as e:
            logger.error(f"Error in merging frame compatibility check: {e}", exc_info=True)
            return False

    def _get_object_position(self, frame: Dict, obj_id: str) -> Optional[List[float]]:
        """Get position of specific object in frame"""
        for metadata in frame['motion_metadata']:
            if metadata['raw_object_id'] == obj_id:
                return metadata['raw_location']
        return None

    def _merge_object_frames(self, frames: List[Dict]) -> Optional[Dict]:
        if not frames:
            return None
        try:
            merged = frames[0].copy()

            # Group annotations by object ID
            object_groups = {}
            for frame in frames:
                for motion_data in frame['motion_metadata']:
                    obj_id = motion_data.get('raw_object_id')
                    if obj_id is None:
                        logger.warning(f"Missing raw_object_id in frame {frame['frame_id']}")
                        continue
                    if obj_id not in object_groups:
                        object_groups[obj_id] = []
                    object_groups[obj_id].append(motion_data)

            # Merge object groups
            merged_metadata = []
            for obj_id, obj_data in object_groups.items():
                if not obj_data:
                    logger.warning(f"No data for object {obj_id}")
                    continue

                confidences = np.array([data.get('sensor_fusion_confidence', 0.5) for data in obj_data])
                confidences = np.maximum(confidences, 0.1)
                weights = confidences / np.sum(confidences)

                # Calculate weighted averages
                positions = np.array([data.get('raw_location', [0, 0, 0]) for data in obj_data])
                avg_position = np.average(positions, weights=weights, axis=0)

                # Use the latest data as base
                latest = max(obj_data, key=lambda x: self._parse_timestamp(x.get('raw_timestamp', '')))
                latest_copy = latest.copy()
                latest_copy['raw_location'] = avg_position.tolist()

                # Add condensation metadata
                latest_copy['condensed_info'] = {
                    'frame_count': len(obj_data),
                    'time_span': self._parse_timestamp(obj_data[-1].get('raw_timestamp', '')) -
                                self._parse_timestamp(obj_data[0].get('raw_timestamp', '')),
                    'position_variance': float(np.var(positions, axis=0).mean()),
                    'average_confidence': float(np.mean(confidences)),
                    'source_frame_ids': [frame['frame_id'] for frame in frames]
                }

                merged_metadata.append(latest_copy)

            merged['motion_metadata'] = merged_metadata
            return merged
        except Exception as e:
            logger.error(f"Error merging frames: {e}", exc_info=True)
            return None