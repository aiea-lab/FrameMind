import numpy as np
from src.core.frame import Frame

class StaticCondenser:
    def __init__(self, frames):
        self.frames = frames
        self.condensed_frame = None

    def combine_static_frames(self):
        combined_data = {}
        for frame in self.frames:
            for slot, value in frame.slots.items():
                if isinstance(value, (int, float)):
                    combined_data.setdefault(slot, []).append(value)
                elif isinstance(value, (list, tuple, np.ndarray)):
                    combined_data.setdefault(slot, []).append(np.array(value))

        # Aggregate combined data, e.g., averaging numeric values
        condensed_data = {}
        for key, values in combined_data.items():
            if isinstance(values[0], np.ndarray):
                condensed_data[key] = np.mean(values, axis=0).tolist()
            else:
                condensed_data[key] = sum(values) / len(values) if len(values) > 0 else None

        self.condensed_frame = Frame(name="CondensedFrame", data=condensed_data)

    def get_condensed_frame(self):
        if self.condensed_frame is None:
            self.combine_static_frames()
        return self.condensed_frame

    def condense_frames(self):
        # Example: Perform condensation logic on frames
        condensed = []
        for frame in self.frames:
            # Condense logic here (example: deduplication, simplification)
            condensed.append(frame)
        return condensed