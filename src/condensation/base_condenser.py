from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np

@dataclass
class CondenserConfig:
    """Base configuration for frame condensation"""
    time_window: float = 0.1  # seconds
    min_confidence: float = 0.7 # sensor data confidence --- for condensation -- Why ?
    max_position_gap: float = 0.5  # meters
    output_dir: Path = Path("output/condensed")

class BaseCondenser(ABC):
    """Base class for frame condensation"""
    
    def __init__(self, config: CondenserConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    @abstractmethod
    def condense_frames(self, frames: List[Dict]) -> List[Dict]:
        """Condense frames based on specific criteria"""
        pass
    
    def save_condensed_frames(self, frames: List[Dict], output_file: Path):
        """Save condensed frames to file"""
        with open(output_file, 'w') as f:
            json.dump(frames, f, indent=2)
            
    def load_frames(self, input_file: Path) -> List[Dict]:
        """Load frames from file"""
        with open(input_file, 'r') as f:
            return json.load(f)
            
    def _calculate_time_difference(self, t1: str, t2: str) -> float:
        """Calculate time difference between timestamps"""
        time1 = float(t1.split('.')[-1])
        time2 = float(t2.split('.')[-1])
        return abs(time1 - time2)