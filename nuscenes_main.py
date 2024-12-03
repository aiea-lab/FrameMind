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
from src.utils.create_frames import create_nuscenes_frames
# from src.processing import frame_processing
from src.processing.frame_processing import *
from src.parsers.scene_processor import process_one_scene
from src.processing.frame_processing import NuScenesParser

# Add new imports for condensation
from src.condensation.config import CondensationConfig
from src.condensation.object_condenser import ObjectCondenser
from src.condensation.sample_condenser import SampleCondenser
from src.condensation.scene_condenser import SceneCondenser

from src.trajectory.config.trajectory_config import TrajectoryConfig
from src.trajectory.analysis.trajectory_analyzer import TrajectoryAnalyzer
from src.trajectory.visualization.trajectory_visualizer import TrajectoryVisualizer



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
    # Example: Select the first scene in the dataset
    scene_token = nusc.scene[0]['token'] #0061 - Boston level
    print(f"Processing scene: {nusc.get('scene', scene_token)['name']}")
    results = process_one_scene(nusc, scene_token)

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
        time_window=0.1,
        min_confidence=0.7,
        max_position_gap=0.5, 
        output_dir=output_dir / "condensed"
    )

    # Initialize condensers
    object_condenser = ObjectCondenser(condenser_config)
    sample_condenser = SampleCondenser(condenser_config)
    scene_condenser = SceneCondenser(condenser_config)

    # Step 6: Perform condensation
    print("Performing frame condensation...")
    
    condensed_objects = object_condenser.condense_frames(object_frames)
    print(f"Object frames condensed: {len(object_frames)} → {len(condensed_objects)}")
    
    condensed_samples = sample_condenser.condense_frames(sample_frames)
    print(f"Sample frames condensed: {len(sample_frames)} → {len(condensed_samples)}")
    
    condensed_scene = scene_condenser.condense_frames([scene_frame])
    print(f"Scene frames processed: 1 → {len(condensed_scene)}")

    # Add this before condensation
    print("\nSample frame example:")
    print(json.dumps(sample_frames[0], indent=2))

    print("\nScene frame structure:")
    print(json.dumps(scene_frame, indent=2))
    
    # Step 7: Save condensed frames
    condensed_dir = output_dir / "condensed"
    condensed_dir.mkdir(exist_ok=True)

    # Save condensed frames with NumpyEncoder
    with open(condensed_dir / 'condensed_object_frames.json', 'w') as f:
        json.dump(condensed_objects, f, cls=NumpyEncoder, indent=2)
    
    with open(condensed_dir / 'condensed_sample_frames.json', 'w') as f:
        json.dump(condensed_samples, f, cls=NumpyEncoder, indent=2)
    
    with open(condensed_dir / 'condensed_scene_frames.json', 'w') as f:
        json.dump(condensed_scene, f, cls=NumpyEncoder, indent=2)

    # Step 8: Save condensation metrics
    metrics = {
        'timestamp': datetime.now().isoformat(),
        'scene_token': scene_token,
        'statistics': {
            'object_frames': {
                'original': len(object_frames),
                'condensed': len(condensed_objects),
                'reduction_ratio': 1 - (len(condensed_objects) / len(object_frames))
            },
            'sample_frames': {
                'original': len(sample_frames),
                'condensed': len(condensed_samples),
                'reduction_ratio': 1 - (len(condensed_samples) / len(sample_frames))
            },
            'scene_frames': {
                'original': 1,
                'condensed': len(condensed_scene)
            }
        }
    }

    with open(condensed_dir / 'condensation_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\nCondensation complete!")
    print(f"Results saved to: {condensed_dir}")
    
    # Print final summary
    print("\nProcessing Summary:")
    print(f"Scene name: {nusc.get('scene', scene_token)['name']}")
    print(f"Object Frames: {metrics['statistics']['object_frames']['reduction_ratio']*100:.1f}% reduction")
    print(f"Sample Frames: {metrics['statistics']['sample_frames']['reduction_ratio']*100:.1f}% reduction")
    print(f"Scene Frames: 1 → {len(condensed_scene)}")

    # Step 9: Initialize trajectory analysis
    print("\nInitializing trajectory analysis...")
    trajectory_config = TrajectoryConfig(
        prediction_horizon=2.0,      # 2 seconds prediction
        min_frames=3,               # minimum frames needed
        confidence_threshold=0.7     # minimum confidence score
    )
    
    trajectory_analyzer = TrajectoryAnalyzer(trajectory_config)
    
    try:
        print("Analyzing trajectories...")
        trajectory_results = trajectory_analyzer.analyze_trajectory(condensed_objects)
        
        # Debug print to understand input structure
        print("\nDebug: First condensed object structure:")
        if condensed_objects:
            print(json.dumps(condensed_objects[0], indent=2))
        
        # Process and analyze trajectories
        trajectory_results = {}
        for obj_frame in condensed_objects:
            try:
                # Extract object ID and analyze trajectory
                obj_id = obj_frame.get('object_id', 'unknown')
                trajectory_data = trajectory_analyzer.analyze_single_object(obj_frame)
                if trajectory_data:
                    trajectory_results[obj_id] = trajectory_data
            except Exception as e:
                print(f"Error processing object frame: {e}")

        # Verify trajectory_results is a dictionary
        if not isinstance(trajectory_results, dict):
            print(f"Warning: Unexpected trajectory results format: {type(trajectory_results)}")
            trajectory_results = {}
        
        # Save trajectory results
        trajectory_dir = output_dir / "trajectory"
        trajectory_dir.mkdir(exist_ok=True)

        with open(trajectory_dir / 'trajectory_analysis.json', 'w') as f:
            json.dump(trajectory_results, f, cls=NumpyEncoder, indent=2)

        # Print analysis summary
        print("\nTrajectory Analysis Summary:")
        print(f"Analyzed trajectories for {len(trajectory_results)} objects")

        # Process each trajectory
        for obj_id, trajectory in trajectory_results.items():
            print(f"\nObject {obj_id}:")
            try:
                # Access trajectory data with error checking
                category = trajectory.get('object_category', 'unknown')
                motion_patterns = trajectory.get('motion_patterns', {})
                predictions = trajectory.get('predictions', {})

                print(f"Category: {category}")
                print(f"Average Speed: {motion_patterns.get('avg_speed', 0):.2f} m/s")
                print(f"Motion Type: "
                    f"{'Stationary' if motion_patterns.get('is_stationary', True) else 'Moving'}, "
                    f"{'Turning' if motion_patterns.get('is_turning', False) else 'Straight'}, "
                    f"{'Accelerating' if motion_patterns.get('is_accelerating', False) else 'Constant Speed'}")
                print(f"Prediction Confidence: {predictions.get('confidence', 0):.2f}")
            except Exception as e:
                print(f"Error processing trajectory for object {obj_id}: {e}")

        print(f"\nTrajectory analysis results saved to: {trajectory_dir}")

    except Exception as e:
        print(f"Error in trajectory analysis: {e}")
        trajectory_results = {}

    # Final summary with error handling
    print("\nFinal Processing Summary:")
    print(f"Scene name: {nusc.get('scene', scene_token)['name']}")
    print(f"Object Frames: {metrics['statistics']['object_frames']['reduction_ratio']*100:.1f}% reduction")
    print(f"Sample Frames: {metrics['statistics']['sample_frames']['reduction_ratio']*100:.1f}% reduction")
    print(f"Scene Frames: 1 → {len(condensed_scene)}")
    print(f"Trajectories Analyzed: {len(trajectory_results)}")


    # In your main function, after trajectory analysis:
    try:
        print("\nGenerating trajectory visualizations...")
        visualizer = TrajectoryVisualizer()
        
        # Create visualization directory
        vis_dir = trajectory_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Generate individual trajectory plots
        for obj_id, trajectory in trajectory_results.items():
            vis_path = vis_dir / f"trajectory_{obj_id}.png"
            visualizer.visualize_trajectory(trajectory, str(vis_path))
        
        # Generate combined visualization
        combined_vis_path = vis_dir / "all_trajectories.png"
        visualizer.visualize_multiple_trajectories(
            trajectory_results, 
            str(combined_vis_path)
        )
        
        print(f"Visualizations saved to: {vis_dir}")

    except Exception as e:
        print(f"Error generating visualizations: {e}")

    
if __name__ == "__main__":
    main()
