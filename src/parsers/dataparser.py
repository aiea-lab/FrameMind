from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path
from elements.cameras import Cameras  # Adjust based on your project structure
from elements.scene_box import SceneBox  # Adjust based on your project structure

@dataclass
class NuScenesDataParserConfig:
    """
    Configuration class for NuScenes data parsing.
    """
    data: Path
    data_dir: Path
    version: str = "v1.0-mini"
    cameras: List[str] = field(default_factory=lambda: [
        "CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_FRONT_LEFT", 
        "CAM_BACK", "CAM_BACK_LEFT", "CAM_BACK_RIGHT"
    ])
    mask_dir: Optional[Path] = None
    train_split_fraction: float = 0.9
    verbose: bool = True