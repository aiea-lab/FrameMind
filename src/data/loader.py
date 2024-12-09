from pathlib import Path
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class NuScenesLoader:
    """Data loader for NuScenes dataset"""
    
    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)
        self.scenes_path = self.data_root / "scenes"
        
    def load_scene(self, scene_token: str) -> Dict:
        """Load a specific scene by token"""
        scene_file = self.scenes_path / f"{scene_token}.json"
        if not scene_file.exists():
            raise FileNotFoundError(f"Scene file not found: {scene_file}")
            
        with open(scene_file, 'r') as f:
            return json.load(f)
            
    def load_object_frames(self, scene_token: str) -> List[Dict]:
        """Load object frames for a scene"""
        frames_path = self.scenes_path / scene_token / "object_frames"
        frames = []
        
        if not frames_path.exists():
            logger.warning(f"No object frames found for scene: {scene_token}")
            return frames
            
        for frame_file in sorted(frames_path.glob("*.json")):
            with open(frame_file, 'r') as f:
                frames.append(json.load(f))
                
        return frames
        
    def load_sample_frames(self, scene_token: str) -> List[Dict]:
        """Load sample frames for a scene"""
        frames_path = self.scenes_path / scene_token / "sample_frames"
        frames = []
        
        if not frames_path.exists():
            logger.warning(f"No sample frames found for scene: {scene_token}")
            return frames
            
        for frame_file in sorted(frames_path.glob("*.json")):
            with open(frame_file, 'r') as f:
                frames.append(json.load(f))
                
        return frames