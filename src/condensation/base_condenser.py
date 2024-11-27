from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np
from .config import CondensationConfig

class BaseCondenser(ABC):
    """Base class for all condensers"""
    
    def __init__(self, config: CondensationConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        """Condense frames based on implementation strategy"""
        pass

    def validate_frame(self, frame: Dict) -> bool:
        """Basic frame validation"""
        try:
            if 'motion_metadata' not in frame:
                return False
            metadata = frame['motion_metadata'][0]
            required = ['timestamp', 'location']
            return all(field in metadata for field in required)
        except Exception:
            return False

    def _calculate_time_difference(self, t1: str, t2: str) -> float:
        """Calculate time difference between timestamps"""
        time1 = float(t1.split('.')[-1])
        time2 = float(t2.split('.')[-1])
        return abs(time2 - time1)

    def _validate_motion_data(self, metadata: Dict) -> bool:
        """Validate motion data"""
        try:
            velocity = np.array(metadata.get('velocity', [0, 0, 0]))
            speed = np.linalg.norm(velocity)
            return speed <= self.config.max_velocity
        except Exception:
            return False