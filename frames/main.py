from frame_structure import FrameManager
from nuscenes_utils import initialize_nuscenes, create_frame_from_sample
# from symbolic_parser import symbolicParser


dataroot = '/Users/ananya/Documents/frames/frames/v1.0-mini'
version = 'v1.0-mini'  # or 'v1.0-trainval' for the full dataset
nusc = initialize_nuscenes(version, dataroot)

frame_manager = FrameManager()

scene = nusc.scene[0]
first_sample_token = scene['first_sample_token']
sample = nusc.get('sample', first_sample_token)

frame_1 = create_frame_from_sample(nusc, sample, frame_manager)

next_sample_token = sample['next']  # Assuming the sample has a 'next' token
if next_sample_token:
    next_sample = nusc.get('sample', next_sample_token)
    frame_2 = create_frame_from_sample(nusc, next_sample, frame_manager)
else:
    frame_2 = None  # In case there is no next sample

# parser = symbolicParser()

# print("Symbolic Representation for Frame 1:")
# symbolic_rep_1 = parser.parse_frame(frame_1)
# for key, value in symbolic_rep_1.items():
#     print(f"{key.name}: {value}")

# print("Symbolic Representation for Frame 1:")
# symbolic_rep_1 = parser.parse_frame(frame_1)
# for key, value in symbolic_rep_1.items():
#     print(f"{key.name}: {value}")

# Output both frames
print("Frame 1:", frame_1)
print("Frame 2:", frame_2)

