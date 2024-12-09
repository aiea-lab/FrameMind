import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List
from pathlib import Path
import seaborn as sns

class TrajectoryVisualizer:
    def __init__(self):
        self.colors = sns.color_palette("husl", 8)
        plt.style.use('seaborn')

    def visualize_all_trajectories(self, trajectory_results: Dict, save_path: str):
        """Visualize all trajectories in a single plot"""
        plt.figure(figsize=(15, 10))
        
        # Create main plot
        ax = plt.gca()
        ax.set_aspect('equal')
        
        # Plot each trajectory
        for obj_id, traj_data in trajectory_results.items():
            positions = np.array(traj_data['trajectory']['positions'])
            category = traj_data['object_info']['category']
            motion_type = traj_data['motion_metrics']['motion_type']
            
            # Plot trajectory path
            color = self.colors[hash(obj_id) % len(self.colors)]
            ax.plot(positions[:, 0], positions[:, 1], '-', 
                   color=color, linewidth=2, alpha=0.7,
                   label=f"{category} ({motion_type})")
            
            # Plot start and end points
            ax.scatter(positions[0, 0], positions[0, 1], marker='o', 
                      color=color, s=100, label='_nolegend_')
            ax.scatter(positions[-1, 0], positions[-1, 1], marker='s', 
                      color=color, s=100, label='_nolegend_')
            
            # Plot predictions if available
            if 'predictions' in traj_data:
                pred_positions = np.array(traj_data['predictions']['predicted_positions'])
                ax.plot(pred_positions[:, 0], pred_positions[:, 1], '--', 
                       color=color, alpha=0.5, linewidth=1)
        
        # Add ego vehicle marker at origin
        ax.scatter([0], [0], marker='*', color='red', s=200, label='Ego Vehicle')
        
        # Customize plot
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_title('All Object Trajectories', fontsize=14, pad=20)
        ax.set_xlabel('X Position (m)', fontsize=12)
        ax.set_ylabel('Y Position (m)', fontsize=12)
        
        # Add legend
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_single_trajectory(self, trajectory_data: Dict, save_path: str):
        """Visualize a single object's trajectory with detailed information"""
        fig = plt.figure(figsize=(15, 10))
        
        # Create subplots
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[0, 0])  # Trajectory path
        ax2 = fig.add_subplot(gs[0, 1])  # Velocity profile
        ax3 = fig.add_subplot(gs[1, :])  # Distance and predictions
        
        # Get data
        positions = np.array(trajectory_data['trajectory']['positions'])
        velocities = np.array(trajectory_data['trajectory']['velocities'])
        timestamps = np.array([np.datetime64(t) for t in trajectory_data['trajectory']['timestamps']])
        relative_times = (timestamps - timestamps[0]).astype('timedelta64[ms]').astype(float) / 1000
        
        # 1. Plot trajectory path
        ax1.plot(positions[:, 0], positions[:, 1], '-o', linewidth=2)
        ax1.scatter([0], [0], color='red', marker='*', s=200, label='Ego Vehicle')
        ax1.set_aspect('equal')
        ax1.set_title('Trajectory Path')
        ax1.set_xlabel('X Position (m)')
        ax1.set_ylabel('Y Position (m)')
        ax1.grid(True)
        
        # 2. Plot velocity profile
        speeds = np.linalg.norm(velocities, axis=1)
        ax2.plot(relative_times, speeds, '-o', label='Speed')
        ax2.set_title('Velocity Profile')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Speed (m/s)')
        ax2.grid(True)
        
        # 3. Plot cumulative distance
        distances = np.cumsum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
        distances = np.insert(distances, 0, 0)
        ax3.plot(relative_times, distances, '-o', label='Distance')
        
        # Add predictions if available
        if 'predictions' in trajectory_data:
            pred_positions = np.array(trajectory_data['predictions']['predicted_positions'])
            pred_times = np.linspace(relative_times[-1], 
                                   relative_times[-1] + trajectory_data['predictions']['prediction_horizon'],
                                   len(pred_positions))
            ax1.plot(pred_positions[:, 0], pred_positions[:, 1], '--', 
                    color='red', alpha=0.5, label='Predicted')
        
        # Add trajectory information
        info_text = (
            f"Category: {trajectory_data['object_info']['category']}\n"
            f"Motion Type: {trajectory_data['motion_metrics']['motion_type']}\n"
            f"Avg Speed: {trajectory_data['motion_metrics']['average_speed']:.2f} m/s\n"
            f"Total Distance: {trajectory_data['motion_metrics']['total_distance']:.2f} m"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8))
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()

    def create_analysis_dashboard(self, trajectory_results: Dict, save_dir: Path):
        """Create a comprehensive analysis dashboard"""
        # Create directory for dashboard
        dashboard_dir = save_dir / "dashboard"
        dashboard_dir.mkdir(exist_ok=True)
        
        # 1. Overall scene visualization
        self.visualize_all_trajectories(trajectory_results, 
                                      str(dashboard_dir / "all_trajectories.png"))
        
        # 2. Individual trajectory analysis
        for obj_id, traj_data in trajectory_results.items():
            self.visualize_single_trajectory(traj_data, 
                                          str(dashboard_dir / f"trajectory_{obj_id}.png"))
        
        # 3. Generate HTML dashboard
        self._generate_html_dashboard(trajectory_results, dashboard_dir)

    def _generate_html_dashboard(self, trajectory_results: Dict, dashboard_dir: Path):
        """Generate HTML dashboard with all visualizations"""
        html_content = """
        <html>
        <head>
            <title>Trajectory Analysis Dashboard</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .container { max-width: 1200px; margin: auto; }
                .overview { margin-bottom: 30px; }
                .trajectory { margin-bottom: 50px; }
                img { max-width: 100%; border: 1px solid #ddd; }
                h2 { color: #333; }
                .metrics { background: #f5f5f5; padding: 15px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trajectory Analysis Dashboard</h1>
                
                <div class="overview">
                    <h2>Overall Scene</h2>
                    <img src="all_trajectories.png">
                </div>
        """
        
        # Add individual trajectories
        for obj_id, traj_data in trajectory_results.items():
            html_content += f"""
                <div class="trajectory">
                    <h2>Object {obj_id}</h2>
                    <div class="metrics">
                        <p>Category: {traj_data['object_info']['category']}</p>
                        <p>Motion Type: {traj_data['motion_metrics']['motion_type']}</p>
                        <p>Average Speed: {traj_data['motion_metrics']['average_speed']:.2f} m/s</p>
                        <p>Total Distance: {traj_data['motion_metrics']['total_distance']:.2f} m</p>
                    </div>
                    <img src="trajectory_{obj_id}.png">
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        with open(dashboard_dir / "dashboard.html", 'w') as f:
            f.write(html_content)