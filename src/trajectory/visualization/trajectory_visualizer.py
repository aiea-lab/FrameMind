import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List

class TrajectoryVisualizer:
    def __init__(self):
        self.colors = {
            'actual': '#1f77b4',     # blue
            'predicted': '#ff7f0e',   # orange
            'uncertainty': '#2ca02c'  # green
        }
        
    def visualize_trajectory(self, trajectory_data: Dict, save_path: str):
        """Create trajectory visualization"""
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Get motion data
        motion_data = trajectory_data.get('motion_data', {})
        predictions = trajectory_data.get('predictions', {})
        
        # Plot actual trajectory if available
        if 'position' in motion_data:
            positions = np.array(motion_data['position'])
            ax.plot(positions[:, 0], positions[:, 1], 
                   color=self.colors['actual'],
                   label='Actual Path',
                   linewidth=2)
            
            # Add start and end markers
            ax.scatter(positions[0, 0], positions[0, 1], 
                      color=self.colors['actual'], 
                      marker='o', s=100, label='Start')
            ax.scatter(positions[-1, 0], positions[-1, 1], 
                      color=self.colors['actual'], 
                      marker='s', s=100, label='End')
        
        # Plot predictions if available
        if 'future_positions' in predictions:
            future_pos = np.array(predictions['future_positions'])
            ax.plot(future_pos[:, 0], future_pos[:, 1], 
                   color=self.colors['predicted'],
                   linestyle='--',
                   label='Predicted Path',
                   linewidth=2)
        
        # Add metadata
        self._add_metadata(ax, trajectory_data)
        
        # Customize plot
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Object Trajectory Analysis')
        ax.legend()
        ax.grid(True)
        
        # Make axes equal for proper scaling
        ax.axis('equal')
        
        # Save plot
        plt.savefig(save_path)
        plt.close()

    def visualize_multiple_trajectories(self, trajectories: Dict[str, Dict], save_path: str):
        """Create visualization for multiple trajectories"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        for obj_id, trajectory in trajectories.items():
            motion_data = trajectory.get('motion_data', {})
            predictions = trajectory.get('predictions', {})
            
            if 'position' in motion_data:
                positions = np.array(motion_data['position'])
                ax.plot(positions[:, 0], positions[:, 1], 
                       label=f'Object {obj_id}',
                       linewidth=2)
                
                # Add end point marker
                ax.scatter(positions[-1, 0], positions[-1, 1], 
                          marker='s', s=100)
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.set_title('Multiple Object Trajectories')
        ax.legend()
        ax.grid(True)
        ax.axis('equal')
        
        plt.savefig(save_path)
        plt.close()
    
    def _add_metadata(self, ax, trajectory_data: Dict):
        """Add metadata to plot"""
        motion_data = trajectory_data.get('motion_data', {})
        
        metadata_text = [
            f"Speed: {motion_data.get('average_speed', 0):.2f} m/s",
            f"Motion: {motion_data.get('motion_type', 'Unknown')}",
            f"Distance: {motion_data.get('total_distance', 0):.2f} m"
        ]
        
        # Add text box with metadata
        text = '\n'.join(metadata_text)
        plt.text(0.02, 0.98, text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))