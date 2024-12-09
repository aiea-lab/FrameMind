import logging
from pathlib import Path
from typing import Dict, List
import argparse
import json

from src.condensation.config import CondensationConfig
from src.condensation.object_condenser import ObjectCondenser
from src.condensation.sample_condenser import SampleCondenser
from src.data.loader import NuScenesLoader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run frame condensation')
    parser.add_argument('--data_root', type=str, required=True,
                       help='Path to NuScenes dataset root')
    parser.add_argument('--output_dir', type=str, required=True,
                       help='Path to output directory')
    parser.add_argument('--scene_token', type=str, required=True,
                       help='Scene token to process')
    return parser.parse_args()

def main():
    args = setup_args()
    
    # Setup paths
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    loader = NuScenesLoader(data_root)
    
    config = CondensationConfig(
        time_window=0.2,
        min_confidence=0.3,
        max_position_gap=2.0,
        output_dir=output_dir
    )
    
    object_condenser = ObjectCondenser(config)
    sample_condenser = SampleCondenser(config)
    
    # Load and process frames
    object_frames = loader.load_object_frames(args.scene_token)
    sample_frames = loader.load_sample_frames(args.scene_token)
    
    logger.info(f"Loaded {len(object_frames)} object frames and {len(sample_frames)} sample frames")
    
    # Condense frames
    condensed_objects = object_condenser.condense_frames(object_frames)
    condensed_samples = sample_condenser.condense_frames(sample_frames)
    
    # Calculate metrics
    object_reduction = 100 * (1 - len(condensed_objects) / len(object_frames)) if object_frames else 0
    sample_reduction = 100 * (1 - len(condensed_samples) / len(sample_frames)) if sample_frames else 0
    
    logger.info(f"Scene: {args.scene_token}")
    logger.info(f"Object Frames: {object_reduction:.1f}% reduction")
    logger.info(f"Sample Frames: {sample_reduction:.1f}% reduction")
    
    # Save results
    results = {
        'scene_token': args.scene_token,
        'metrics': {
            'object_reduction': object_reduction,
            'sample_reduction': sample_reduction,
            'original_object_frames': len(object_frames),
            'condensed_object_frames': len(condensed_objects),
            'original_sample_frames': len(sample_frames),
            'condensed_sample_frames': len(condensed_samples)
        },
        'condensed_objects': condensed_objects,
        'condensed_samples': condensed_samples
    }
    
    output_file = output_dir / f"{args.scene_token}_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {output_file}")

if __name__ == '__main__':
    main()