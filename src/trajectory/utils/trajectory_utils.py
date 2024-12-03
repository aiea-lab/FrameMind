import numpy as np
from typing import List, Dict, Tuple

def calculate_time_difference(t1: str, t2: str) -> float:
    """Calculate time difference between timestamps"""
    time1 = float(t1.split('.')[-1])
    time2 = float(t2.split('.')[-1])
    return abs(time2 - time1)

def smooth_trajectory(positions: np.ndarray, window_size: int = 3) -> np.ndarray:
    """Apply smoothing to trajectory positions"""
    return np.array([
        np.mean(positions[max(0, i-window_size):min(len(positions), i+window_size+1)], axis=0)
        for i in range(len(positions))
    ])

def calculate_curvature(positions: np.ndarray) -> np.ndarray:
    """Calculate trajectory curvature"""
    dx_dt = np.gradient(positions[:, 0])
    dy_dt = np.gradient(positions[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    
    curvature = np.abs(dx_dt * d2y_dt2 - dy_dt * d2x_dt2) / \
                (dx_dt * dx_dt + dy_dt * dy_dt)**1.5
    
    return curvature

def interpolate_trajectory(positions: np.ndarray, 
                         timestamps: np.ndarray, 
                         target_dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Interpolate trajectory to uniform time steps"""
    from scipy.interpolate import interp1d
    
    # Create interpolation function
    interp_func = interp1d(timestamps, positions, axis=0, kind='cubic')
    
    # Create new timestamps
    new_timestamps = np.arange(timestamps[0], timestamps[-1], target_dt)
    
    # Interpolate positions
    new_positions = interp_func(new_timestamps)
    
    return new_positions, new_timestamps