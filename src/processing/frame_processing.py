import os
import json
from pathlib import Path
from datetime import datetime
from nuscenes import NuScenes
from src.core.frame_manager import FrameManager
from src.parsers.nuscenes_parser_config import NuScenesDataParserConfig
# from src.condensing.static_condenser import StaticCondenser
from src.parsers.numpy_encoder import NumpyEncoder
from src.parsers.nuscenes_parser import NuScenesParser
import numpy as np


def process_and_condense_frames(dataroot: str, version: str = "v1.0-mini"):
    """Process frames, update trajectories, and apply condensation."""
    # Initialize NuScenes, FrameManager, and configuration
    nusc = NuScenes(version=version, dataroot=dataroot, verbose=True)
    frame_manager = FrameManager()
    config = NuScenesDataParserConfig(
        data=Path(dataroot),
        data_dir=Path(dataroot),
        version=version
    )

    # Parse scenes
    parser = NuScenesParser(config)
    all_scene_outputs = parser.parse_all_scenes()

    # Update trajectories
    frame_manager.update_trajectories()

    # Example of getting a specific object trajectory
    object_id = "1ddf7a48af3d43be8763f154974f09ce"
    try:
        trajectory = frame_manager.get_object_trajectory(object_id)
        print(f"Trajectory for object {object_id}:", trajectory)
    except ValueError as e:
        print(e)

    # Condense frames
    condenser = StaticCondenser(frame_manager.frames)  # Pass the frames to the StaticCondenser
    condensed_frames = condenser.condense_frames()

    # Output condensed frames to JSON
    output_file_path = "condensed_frames_output.json"
    with open(output_file_path, 'w') as outfile:
        json.dump([frame.to_dict() for frame in condensed_frames], outfile, indent=4, cls=NumpyEncoder)

    print(f"Condensed frames have been written to {output_file_path}")

    return all_scene_outputs, condensed_frames

def convert_numpy_types(data):
    """Convert numpy types to Python native types"""
    if isinstance(data, dict):
        return {key: convert_numpy_types(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_numpy_types(item) for item in data]
    elif isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64)):
        return int(data)
    elif isinstance(data, (np.float_, np.float16, np.float32, np.float64)):
        return float(data)
    elif isinstance(data, np.bool_):
        return bool(data)
    return data

def save_object_frames(object_frames, output_file):
    converted_frames = convert_numpy_types(object_frames)
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output file path
    output_file = os.path.join(output_dir, "object_frames.json")
    
    # Save the object_frames as a JSON file
    with open(output_file, "w") as f:
        json.dump(converted_frames, f, indent=4)
    
    print(f"Object frames saved to {output_file}")

def save_scene_frames(scene_frames, output_file):
    # Ensure the output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output file path
    output_file = os.path.join(output_dir, "scene_frames.json")
    
    # Save the scene_frames as a JSON file
    with open(output_file, "w") as f:
        json.dump(scene_frames, f, indent=4)
    
    print(f"Scene frames saved to {output_file}")

def save_sample_frames(sample_frames, output_file):
    # Ensure the output directory exists
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define the output file path
    output_file = os.path.join(output_dir, "sample_frames.json")
    
    # Save the sample_frames as a JSON file
    with open(output_file, "w") as f:
        json.dump(sample_frames, f, indent=4)
    
    print(f"Sample frames saved to {output_file}")