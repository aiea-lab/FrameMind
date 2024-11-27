import sys
from pathlib import Path
import os
import json
import random
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
    

    # # Step 5: Process and condense frames
    # print("Processing and condensing frames...")
    # _, condensed_frames = process_and_condense_frames(dataroot, version)

    # # Save processed and condensed frames
    # with open(output_dir / 'condensed_frames_output.json', 'w') as f:
    #     json.dump([frame.to_dict() for frame in condensed_frames], f, indent=4)
    # print("Condensed frames output saved to condensed_frames_output.json")

    # # Step 6: Save parsed scenes output
    # with open(output_dir / 'parsed_scenes_output.json', 'w') as f:
    #     json.dump(all_scene_outputs, f, indent=4)
    # print("Parsed scenes output saved to parsed_scenes_output.json")

    # print("Processing complete!")

if __name__ == "__main__":
    main()
