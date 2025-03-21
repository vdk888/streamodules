"""
Utility functions for the trading platform
"""

import json
import math
import datetime
try:
    import numpy as np
except ImportError:
    np = None

# Custom JSON encoder to handle NaN, Infinity, etc.
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if np is not None and isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif np is not None and isinstance(obj, (np.floating, np.float64, np.float32)):
            if math.isnan(obj) or math.isinf(obj):
                return None
            return float(obj)
        elif np is not None and isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

# Helper function to clean data for JSON serialization
def clean_for_json(obj):
    """Clean object for JSON serialization by handling numpy types, NaN, Infinity, etc."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(i) for i in obj]
    elif np is not None and isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (float, int)) or (np is not None and isinstance(obj, (np.floating, np.float64, np.float32))):
        if isinstance(obj, float) or (np is not None and isinstance(obj, (np.floating, np.float64, np.float32))):
            obj_float = float(obj)
            return None if math.isnan(obj_float) or math.isinf(obj_float) else obj_float
        return obj
    elif np is not None and isinstance(obj, np.ndarray):
        return clean_for_json(obj.tolist())
    elif isinstance(obj, datetime.datetime):
        return obj.isoformat()
    return obj