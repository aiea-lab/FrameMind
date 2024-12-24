import sys
from pathlib import Path
import os
import json
import random
import matplotlib.pyplot as plt
from datetime import datetime
from nuscenes import NuScenes
from src.core.coordinate import Coordinate
from src.core.frame_manager import FrameManager
from src.core.status import Status
from datetime import datetime
from dataclasses import dataclass, field
from typing import List, Dict, Optional
# from src.condensing.static_condenser import StaticCondenser
from src.parsers.numpy_encoder import NumpyEncoder
from src.parsers.scene_elements import Cameras, SceneBox
from src.elements.cameras import Cameras
from src.elements.scene_box import SceneBox
from src.elements.camera_data import CameraData
from src.elements.sample import Sample
from src.core.scene import Scene
from src.core.nuscenes_database import NuScenesDatabase
from src.parsers.nuscenes_parser_config import NuScenesDataParserConfig
from src.core.frame_manager import FrameManager
from src.parsers.numpy_encoder import NumpyEncoder
from src.parsers.nuscenes_parser_config import NuScenesDataParserConfig
# from src.processing.frame_processing import process_and_condense_frames
from src.parsers.numpy_encoder import NumpyEncoder
from src.trajectory.generator import TrajectoryGenerator
from src.utils.create_frames import create_nuscenes_frames
# from src.processing import frame_processing
from src.processing.frame_processing import *
from src.parsers.scene_processor import process_one_scene
from src.processing.frame_processing import NuScenesParser

# Add new imports for condensation
from src.condensation.config import CondensationConfig, CondensationParams
from src.condensation.object_condenser import ObjectCondenser
from src.condensation.sample_condenser import SampleCondenser
from src.condensation.scene_condenser import SceneCondenser
from src.condensation.metrics.confidence import ConfidenceMetrics
from src.condensation.output_generator import CondensedOutputGenerator
from src.condensation.config import CondensationParams

from src.trajectory.config.trajectory_config import TrajectoryConfig
from src.trajectory.analysis.trajectory_analyzer import TrajectoryAnalyzer
from src.trajectory.visualization.trajectory_visualizer import TrajectoryVisualizer
from src.trajectory.visualization.dashboard_generator import DashboardGenerator
from src.trajectory.generator import TrajectoryGenerator

from src.anomaly.binary_anomaly_detector import BinaryTrajectoryAnomalyDetector, detect_trajectory_anomalies


# from logger import get_logger
sys.path.append(str(Path(__file__).resolve().parent.parent))
sys.path.append(str(Path(__file__).resolve().parent.parent / "src"))
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

sys.path.append('/Users/ananya/Desktop/frames/data/raw/nuscenes')

def main():

    # Configure paths and version
    dataroot = '/Applications/FrameMind/data/raw/nuscenes/v1.0-mini'
    version = "v1.0-mini"

    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)

    # Create condensed directory right away
    condensed_dir = output_dir / "condensed"
    condensed_dir.mkdir(exist_ok=True)

    # Initialize configuration
    config = NuScenesDataParserConfig(
        data=Path(dataroot),
        data_dir=Path(dataroot),
        version=version
    )

    #Initialize NuScenes to get a scene token (NUSCENE_API)
    nusc = NuScenes(version=config.version, dataroot=str(config.data_dir))

    # Step 1: Create frames
    print("Creating frames...")
    frame_manager = create_nuscenes_frames(dataroot, version)

    # Step 2: Parse all scenes using NuScenesParser
    print("Parsing all scenes...")
    parser = NuScenesParser(config)
    all_scene_outputs = parser.parse_all_scenes()

    # Step 2: Select a Scene
    # Example: Select the third scene in the dataset
    scene_token = nusc.scene[4]['token'] #scene-0655 - Parking lot, parked cars, jaywalker, be... [18-08-27 15:51:32]   20s, boston-seaport, #anns:2332
    print(f"Processing scene: {nusc.get('scene', scene_token)['name']}")
    

    # Step 3: Process the Selected Scene
    results = process_one_scene(nusc, scene_token)

    # Step 4: Handle Outputs
    scene_frame = results['scene_frame']
    sample_frames = results['sample_frames']
    object_frames = results['object_frames']

    # Save Outputs to JSON Files
    save_object_frames(object_frames, output_dir / 'object_frames.json')
    save_sample_frames(sample_frames, output_dir / 'sample_frames.json')
    save_scene_frames(scene_frame,  output_dir / 'scene_frame.json' )

    # Step 5: Initialize condensation
    print("\nInitializing frame condensation...")
    condenser_config = CondensationConfig(
        time_window=0.2,
        min_confidence=0.3,
        max_position_gap=2.0,
        output_dir=output_dir / "condensed"
    )

    confidence_metrics = ConfidenceMetrics()
    object_condenser = ObjectCondenser(condenser_config)

    print("\nProcessing frames with confidence metrics...")
    for frame in object_frames:
        if 'motion_metadata' in frame:
            for metadata in frame['motion_metadata']:
                confidence = confidence_metrics.calculate_comprehensive_confidence({
                    'raw_num_lidar_points': metadata.get('raw_num_lidar_points', 0),
                    'raw_num_radar_points': metadata.get('raw_num_radar_points', 0),
                    'derived_distance_to_ego': metadata.get('derived_distance_to_ego', 100),
                    'derived_visibility': metadata.get('derived_visibility', 0),
                    'derived_occlusion_status': metadata.get('derived_occlusion_status', 'Partially Occluded')
                })
                metadata['derived_sensor_fusion_confidence'] = confidence

    # Perform condensation
    print("\nPerforming frame condensation...")
    condensed_objects = object_condenser.condense_frames(object_frames)

    # Generate structured output
    params = CondensationParams(
        time_window=0.2,
        min_confidence=0.3,
        max_position_gap=2.0
    )

    output_generator = CondensedOutputGenerator(
        params=params,
        nusc=nusc,  # Pass your nuScenes instance
        scene_token=scene_token  # Pass your scene token
        )
    
    # Generate structured condensed output
    print("\nGenerating structured condensed output...")
    condensed_output = output_generator.generate_condensed_output(
        original_frames=object_frames,
        condensed_frames=condensed_objects
    )

    # Save structured output
    print("\nSaving structured output...")
    output_paths = {
        'structured': condensed_dir / 'structured_condensed_output.json',
        'raw_condensed': condensed_dir / 'condensed_object_frames.json',
        'metrics': condensed_dir / 'condensation_metrics.json'
    }

    # Save all outputs
    for name, path in output_paths.items():
        with open(path, 'w') as f:
            if name == 'structured':
                json.dump(condensed_output, f, cls=NumpyEncoder, indent=2)
            elif name == 'raw_condensed':
                json.dump(condensed_objects, f, cls=NumpyEncoder, indent=2)
            elif name == 'metrics':
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'scene_token': scene_token,
                    'statistics': condensed_output['statistics']
                }, f, cls=NumpyEncoder, indent=2)

    # Print enhanced summary
    print("\nCondensation Summary:")
    print(f"Scene name: {nusc.get('scene', scene_token)['name']}")
    print(f"Original frames: {condensed_output['meta_info']['num_original_frames']}")
    print(f"Condensed frames: {condensed_output['meta_info']['num_condensed_frames']}")
    print(f"Reduction ratio: {condensed_output['statistics']['frame_reduction_ratio']*100:.1f}%")
    print(f"Average tracking confidence: {condensed_output['statistics']['avg_tracking_confidence']:.2f}")
    print(f"\nResults saved to: {condensed_dir}")

    print("\nGenerating trajectories...")
    trajectory_generator = TrajectoryGenerator(
        min_points=3,
        prediction_horizon=2.0
    )

    # Generate trajectories
    trajectory_generator = TrajectoryGenerator(min_points=3, prediction_horizon=2.0)
    trajectory_results = trajectory_generator.generate_trajectories(condensed_output)

    # Offset positions relative to ego vehicle
    ego_vehicle_global_position = [0, 0, 0]   # Example: ego vehicle's position
    trajectories = trajectory_generator.process_positions(trajectory_results, ego_vehicle_global_position)
  
    
    # Optional: Separate or scale trajectories for debugging
    offset = 100
    for i, (obj_id, traj) in enumerate(trajectories.items()):
        positions = np.array(traj['trajectory']['positions'])
        positions[:, 0] += i * offset  # Separate objects in X direction for better visualization
        traj['trajectory']['positions'] = positions.tolist()

    # Save trajectory results
    trajectory_dir = output_dir / "trajectory"
    trajectory_dir.mkdir(exist_ok=True)

    with open(trajectory_dir / 'trajectory_analysis.json', 'w') as f:
        json.dump(trajectory_results, f, cls=NumpyEncoder, indent=2)

    # Print trajectory summary
    print("\nTrajectory Analysis Summary:")
    print(f"Analyzed trajectories for {len(trajectory_results)} objects")

    # Step 7: Generate visualization and dashboard
    print("\nGenerating dashboard and visualizations...")
    dashboard_generator = DashboardGenerator()

   # Generate overview plot
    overview_plot_path = trajectory_dir / "overview.png"
    dashboard_generator.create_overview_plot(trajectory_results, str(overview_plot_path))
    print(f"Overview plot saved to: {overview_plot_path}")

    # Generate HTML dashboard
    dashboard_dir = trajectory_dir / "html"
    dashboard_dir.mkdir(exist_ok=True)
    metrics = {obj_id: trajectory['motion_metrics'] for obj_id, trajectory in trajectory_results.items()}
    dashboard_generator.generate_html_dashboard(trajectory_results, metrics, dashboard_dir)
    print(f"Dashboard saved to: {dashboard_dir / 'dashboard.html'}")

    # Step 8: Visualizations
    visualizer = TrajectoryVisualizer()
    vis_dir = trajectory_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    visualizer.create_analysis_dashboard(trajectory_results, vis_dir)
    print(f"Visualizations saved to: {vis_dir}")

    for obj_id, trajectory in trajectory_results.items():
        print(f"\nObject {obj_id}:")
        print(f"Category: {trajectory['object_info']['category']}")
        print(f"Motion type: {trajectory['motion_metrics']['motion_type']}")
        print(f"Average speed: {trajectory['motion_metrics']['average_speed']:.2f} m/s")
        print(f"Total distance: {trajectory['motion_metrics']['total_distance']:.2f} m")

    # Generating trajectory sections for HTML
    print("\nGenerating trajectory HTML sections...")
    trajectory_html_dir = trajectory_dir / "html"
    trajectory_html_dir.mkdir(exist_ok=True)
     
    # After generating trajectories
    print("\nGenerating trajectory visualizations...")
    visualizer = TrajectoryVisualizer()

    # Create visualization directory
    vis_dir = output_dir / "trajectory" / "visualizations"
    vis_dir.mkdir(exist_ok=True, parents=True)

    # Generate comprehensive dashboard
    visualizer.create_analysis_dashboard(trajectory_results, vis_dir)

    print(f"\nVisualizations saved to: {vis_dir}")
    
    print("\nPerforming anomaly detection...")
    anomaly_results = detect_trajectory_anomalies(trajectory_results, contamination=0.1)

    # Save anomaly results
    anomaly_dir = output_dir / "anomalies"
    anomaly_dir.mkdir(exist_ok=True)
    with open(anomaly_dir / 'binary_anomaly_results.json', 'w') as f:
        json.dump(anomaly_results, f, cls=NumpyEncoder, indent=2)

    # Print anomaly summary
    print("\nAnomaly Detection Summary:")
    print(f"Total objects analyzed: {anomaly_results['statistics']['total_objects']}")
    print(f"Anomalous objects detected: {anomaly_results['statistics']['anomalous_objects']}")
    print(f"Anomaly percentage: {anomaly_results['statistics']['anomaly_percentage']:.1f}%")

    # Add anomaly information to trajectory_results
    for obj_id in trajectory_results:
        if obj_id in anomaly_results['object_anomalies']:
            trajectory_results[obj_id]['is_anomalous'] = \
                anomaly_results['object_anomalies'][obj_id]['is_anomalous']

    # Continue with your existing visualization code, but update to include anomaly info
    print("\nGenerating dashboard and visualizations...")
    dashboard_generator = DashboardGenerator()

    # Update your metrics dictionary to include anomaly information
    metrics = {
        obj_id: {
            **trajectory['motion_metrics'],
            'is_anomalous': trajectory.get('is_anomalous', 0)
        }
        for obj_id, trajectory in trajectory_results.items()
    }

    # Your existing visualization code continues...
    dashboard_dir = trajectory_dir / "html"
    dashboard_dir.mkdir(exist_ok=True)
    dashboard_generator.generate_html_dashboard(trajectory_results, metrics, dashboard_dir)

    # Update your object printing to include anomaly status
    for obj_id, trajectory in trajectory_results.items():
        print(f"\nObject {obj_id}:")
        print(f"Category: {trajectory['object_info']['category']}")
        print(f"Motion type: {trajectory['motion_metrics']['motion_type']}")
        print(f"Average speed: {trajectory['motion_metrics']['average_speed']:.2f} m/s")
        print(f"Total distance: {trajectory['motion_metrics']['total_distance']:.2f} m")
        print(f"Anomaly status: {'ANOMALOUS' if trajectory.get('is_anomalous', 0) else 'NORMAL'}")

    
    
if __name__ == "__main__":
    main()
