from frame_structure import FrameManager
from nuscenes_utils import initialize_nuscenes, create_frame_from_sample

dataroot = '/Users/ananya/opt/anaconda3/lib/python3.9/site-packages'
version = 'v1.0-mini'  # or 'v1.0-trainval' for the full dataset
nusc = initialize_nuscenes(version, dataroot)

frame_manager = FrameManager()

scene = nusc.scene[0]
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

frame = create_frame_from_sample(nusc, sample, frame_manager)

print(frame)