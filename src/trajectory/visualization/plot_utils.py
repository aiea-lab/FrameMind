import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from typing import Dict
from ..models.data_models import TrajectoryData, ObjectInfo

class PlotUtils:
    """Utility class for common plotting functions"""
    
    @staticmethod
    def setup_axis_properties(ax: plt.Axes, title: str, xlabel: str, ylabel: str):
        """Set up common axis properties"""
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
    
    @staticmethod
    def add_ego_vehicle_marker(ax: plt.Axes):
        """Add ego vehicle marker to plot"""
        ax.scatter([0], [0], color='red', marker='*', s=200, label='Ego Vehicle')
