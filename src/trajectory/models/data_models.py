from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np
from datetime import datetime

@dataclass
class ObjectInfo:
    """Data class for storing object information"""
    category: str
    id: str
    dimensions: Optional[Tuple[float, float, float]] = None
    confidence: float = 1.0
    
@dataclass
class TrajectoryData:
    """Data class for storing trajectory information"""
    positions: np.ndarray
    velocities: np.ndarray
    timestamps: np.ndarray
    object_info: ObjectInfo
    predictions: Optional[Dict] = None
    uncertainties: Optional[np.ndarray] = None