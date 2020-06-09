import json
import numpy as np 

class NumpyEncoder(json.JSONEncoder):
    """An encoder class for the JSON library.
       This one adds support for JSONifying numpy arrays.
    """
    def default(self, obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        
        return json.JSONEncoder.default(self, obj)