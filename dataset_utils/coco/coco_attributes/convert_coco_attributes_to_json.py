"""This is a quick script to convert the Python 2 joblib file for coco attr
to a json. This script has to be run in Python 2, unfortunately.

The whole reason for doing this is so that other scripts can read this (python3)
json and use it that way.
"""

from sklearn.externals import joblib

import numpy as np
import json

attr_json = "/media/alex/Database/Datasets/COCO/cocoattributes.json"
joblib_attributes_file = "/media/alex/Database/Datasets/COCO/cocottributes_eccv_version.jbl"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


attr = joblib.load(joblib_attributes_file)

json.dump(dict(attr), open(attr_json, "w"), cls=NumpyEncoder)
