import matplotlib.pyplot as plt
import numpy as np
from typing import Dict
from ..models.data_models import TrajectoryData, ObjectInfo
from .plot_utils import PlotUtils

class SingleTrajectoryVisualizer:
    """Handles visualization of individual trajectories"""
    
    def __init__(self):
        self.style_config = {
            'figure.figsize': (15, 12),
            'axes.grid': True,
            'grid.alpha': 0.3,
            'lines.linewidth': 2,
            'font.size': 10
        }
    
    def create_visualization(self, trajectory: TrajectoryData, 
                           metrics: Dict, save_path: str):
        """Create detailed single trajectory visualization"""
        with plt.style.context('seaborn'):
            fig = plt.figure(figsize=self.style_config['figure.figsize'])
            gs = fig.add_gridspec(3, 2, height_ratios=[2, 1, 1])
            
            ax_path = fig.add_subplot(gs[0, 0])
            self._plot_trajectory_path(ax_path, trajectory)
            
            ax_bev = fig.add_subplot(gs[0, 1])
            self._plot_birds_eye_view(ax_bev, trajectory)
            
            ax_vel = fig.add_subplot(gs[1, :])
            self._plot_velocity_profile(ax_vel, trajectory)
            
            ax_heading = fig.add_subplot(gs[2, :])
            self._plot_heading_profile(ax_heading, trajectory)
            
            self._add_metrics_summary(fig, metrics, trajectory.object_info)
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
    
    def _plot_trajectory_path(self, ax: plt.Axes, trajectory: TrajectoryData):
        """Plot trajectory path with uncertainty ellipses"""
        positions = trajectory.positions
        ax.plot(positions[:, 0], positions[:, 1], '-o', label='Actual Path')
        
        if trajectory.uncertainties is not None:
            for pos, uncert in zip(positions[::5], trajectory.uncertainties[::5]):
                ellipse = Ellipse(pos, uncert[0], uncert[1], 
                                angle=np.rad2deg(uncert[2]),
                                alpha=0.2, color='gray')
                ax.add_patch(ellipse)
        
        if trajectory.predictions:
            pred_positions = trajectory.predictions['predicted_positions']
            ax.plot(pred_positions[:, 0], pred_positions[:, 1], '--', 
                   color='red', alpha=0.5, label='Predicted Path')
        
        PlotUtils.setup_axis_properties(ax, 'Trajectory Path', 
                                      'X Position (m)', 'Y Position (m)')
        ax.set_aspect('equal')
        ax.legend()
    
    def _plot_birds_eye_view(self, ax: plt.Axes, trajectory: TrajectoryData):
        """Plot bird's eye view with object dimensions"""
        last_pos = trajectory.positions[-1]
        if trajectory.object_info.dimensions:
            l, w, h = trajectory.object_info.dimensions
            rect = plt.Rectangle((last_pos[0] - l/2, last_pos[1] - w/2), 
                               l, w, angle=0, fill=True, alpha=0.5)
            ax.add_patch(rect)
        
        PlotUtils.add_ego_vehicle_marker(ax)
        PlotUtils.setup_axis_properties(ax, "Bird's Eye View", 
                                      'X Position (m)', 'Y Position (m)')
        ax.set_aspect('equal')
        
        margin = 10
        ax.set_xlim(last_pos[0] - margin, last_pos[0] + margin)
        ax.set_ylim(last_pos[1] - margin, last_pos[1] + margin)
    
    def _plot_velocity_profile(self, ax: plt.Axes, trajectory: TrajectoryData):
        """Plot velocity and acceleration profiles"""
        times = (trajectory.timestamps - trajectory.timestamps[0]).astype('timedelta64[ms]').astype(float) / 1000
        speeds = np.linalg.norm(trajectory.velocities, axis=1)
        
        accelerations = np.diff(speeds) / np.diff(times)
        acc_times = times[:-1] + np.diff(times)/2
        
        ax1 = ax
        ax1.plot(times, speeds, '-', label='Speed', color='blue')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Speed (m/s)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        ax2 = ax1.twinx()
        ax2.plot(acc_times, accelerations, '--', label='Acceleration', 
                color='red', alpha=0.7)
        ax2.set_ylabel('Acceleration (m/s²)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Speed and Acceleration Profiles')
        
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    def _plot_heading_profile(self, ax: plt.Axes, trajectory: TrajectoryData):
        """Plot heading changes over time"""
        times = (trajectory.timestamps - trajectory.timestamps[0]).astype('timedelta64[ms]').astype(float) / 1000
        positions_diff = np.diff(trajectory.positions, axis=0)
        heading_angles = np.rad2deg(np.arctan2(positions_diff[:, 1], 
                                             positions_diff[:, 0]))
        heading_times = times[:-1] + np.diff(times)/2
        
        ax.plot(heading_times, heading_angles, '-o')
        PlotUtils.setup_axis_properties(ax, 'Heading Profile', 
                                      'Time (s)', 'Heading Angle (degrees)')
    
    def _add_metrics_summary(self, fig: plt.Figure, metrics: Dict, 
                           object_info: ObjectInfo):
        """Add metrics summary to the figure"""
        metrics_text = (
            f"Object Info:\n"
            f"  Category: {object_info.category}\n"
            f"  ID: {object_info.id}\n"
            f"  Confidence: {object_info.confidence:.2f}\n\n"
            f"Motion Metrics:\n"
            f"  Motion Type: {metrics['motion_type']}\n"
            f"  Avg Speed: {metrics['average_speed']:.2f} m/s\n"
            f"  Max Speed: {metrics['max_speed']:.2f} m/s\n"
            f"  Total Distance: {metrics['total_distance']:.2f} m\n"
            f"  Straight Line Distance: {metrics['straight_line_distance']:.2f} m\n"
            f"  Duration: {metrics['duration']:.2f} s"
        )
        plt.figtext(1.02, 0.5, metrics_text, fontsize=10,
                   bbox=dict(facecolor='white', alpha=0.8))