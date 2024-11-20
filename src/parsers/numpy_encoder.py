import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)
    
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, Coordinate):
            return [obj.x, obj.y, obj.z]
        elif isinstance(obj, np.generic):
            return obj.item()  # Handle numpy types
        elif hasattr(obj, "__dict__"):
            return obj.__dict__  # Serialize custom objects with __dict__
        return super().default(obj)