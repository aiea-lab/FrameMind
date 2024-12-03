import json
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()  # Convert ndarray to list
        if isinstance(obj, np.generic):
            return obj.item()  # Convert NumPy scalar to Python scalar
        return super().default(obj)
    
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