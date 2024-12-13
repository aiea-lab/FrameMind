import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict
import numpy as np
from ..models.data_models import TrajectoryData
from .plot_utils import PlotUtils

class DashboardGenerator:
    """Handles generation of the visualization dashboard"""
    
    def __init__(self):
        self.colors = sns.color_palette("husl", 8)
    
    def create_overview_plot(self, trajectories: Dict[str, TrajectoryData], save_path: str):
        """Create overview visualization of all trajectories"""
        with plt.style.context('seaborn'):
            plt.figure(figsize=(15, 10))
            ax = plt.gca()
            ax.set_aspect('equal')
            
            for obj_id, trajectory in trajectories.items():
                color = self.colors[hash(obj_id) % len(self.colors)]
                
                if isinstance(trajectory, dict):
                    positions = np.array(trajectory.get('positions', []))
                    if positions.size == 0:
                        print(f"Warning: Skipping object {obj_id} - 'positions' key is missing or empty.")
                        continue
                    category = trajectory.get('object_info', {}).get('category', 'Unknown')
                else:
                    positions = trajectory.positions
                    category = trajectory.object_info.category
                
                ax.plot(positions[:, 0], positions[:, 1], '-', color=color, linewidth=2, alpha=0.7, label=f"{category}")
                ax.scatter(positions[0, 0], positions[0, 1], marker='o', color=color, s=100, label="Start")
                ax.scatter(positions[-1, 0], positions[-1, 1], marker='s', color=color, s=100, label="End")
            
            PlotUtils.add_ego_vehicle_marker(ax)
            PlotUtils.setup_axis_properties(ax, 'All Object Trajectories', 'X Position (m)', 'Y Position (m)')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            plt.close()
    
    def generate_html_dashboard(self, trajectories: Dict[str, TrajectoryData], 
                                metrics: Dict, dashboard_dir: Path):
        """Generate HTML dashboard with interactive features"""
        html_content = self._generate_html_template(trajectories, metrics)
        
        with open(dashboard_dir / "dashboard.html", 'w') as f:
            f.write(html_content)
    
    def _generate_html_template(self, trajectories: Dict[str, TrajectoryData], 
                                metrics: Dict) -> str:
        """Generate HTML template for dashboard"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Advanced Trajectory Analysis Dashboard</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
                .container {{ max-width: 1400px; margin: auto; }}
                .overview {{ margin-bottom: 30px; background: white; padding: 20px; 
                           border-radius: 10px; }}
                .trajectory {{ margin-bottom: 30px; background: white; padding: 20px;
                             border-radius: 10px; }}
                img {{ max-width: 100%; border-radius: 5px; }}
                h2 {{ color: #2c3e50; }}
                .metrics {{ background: #eef2f7; padding: 15px; border-radius: 5px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Trajectory Analysis Dashboard</h1>
                
                <div class="overview">
                    <h2>Overall Scene</h2>
                    <img src="overview.png">
                </div>
                
                {self.generate_trajectory_sections(trajectories, metrics)}
            </div>
        </body>
        </html>
        """
    
    def generate_trajectory_sections(self, trajectories: Dict[str, TrajectoryData],
                                    metrics: Dict) -> str:
        """Generate HTML sections for each trajectory with validation for missing data."""
        sections = []
        for obj_id, trajectory in trajectories.items():
            # Check for missing or empty 'positions'
            if not hasattr(trajectory, 'positions') or not trajectory.positions:
                print(f"Warning: Skipping object {obj_id} - 'positions' key is missing or empty.")
                continue

            # Retrieve metrics safely
            metric = metrics.get(obj_id, {})
            
            # Use placeholders for missing metric fields
            motion_type = metric.get('motion_type', 'Unknown')
            speed = metric.get('speed', 'N/A')
            direction = metric.get('direction', 'N/A')

            sections.append(f"""
                <div class="trajectory">
                    <h2>Object {obj_id}</h2>
                    <div class="metrics">
                        <p>Category: {trajectory.object_info.category}</p>
                        <p>Motion Type: {motion_type}</p>
                        <p>Speed: {speed}</p>
                        <p>Direction: {direction}</p>
                    </div>
                    <div class="trajectory-data">
                        <p>Start Point: {trajectory.start_point}</p>
                        <p>End Point: {trajectory.end_point}</p>
                        <p>Duration: {trajectory.duration}</p>
                    </div>
                </div>
            """)
        return "\n".join(sections)