from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from .config import CondensationConfig
import logging

logger = logging.getLogger(__name__)

class BaseCondenser(ABC):
    """Base class for frame condensation"""
    
    def __init__(self, config: CondensationConfig):
        self.config = config
    
    @abstractmethod
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        """Condense multiple frames into a reduced set"""
        pass
        
    def _calculate_time_difference(self, time1: float, time2: float) -> float:
        """Calculate absolute time difference between frames"""
        return abs(float(time1) - float(time2))
        
    def save_results(self, frames: List[Dict], filename: str) -> None:
        """Save condensed frames to output directory"""
        if self.config.output_dir is None:
            logger.warning("No output directory specified, skipping save")
            return
            
        output_path = self.config.output_dir / filename
        with open(output_path, 'w') as f:
            json.dump(frames, f, indent=2)