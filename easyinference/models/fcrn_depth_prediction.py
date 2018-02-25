
import numpy as np

from easyinference import load_utils



class FCRNDepthPredictor:
    def __init__(self, model_bytes):
        self.graph, self.session = load_utils.parse_tf_model(model_bytes)

    @staticmethod
    def from_path(self, model_path):
        model = load_utils.load_tf_model(model_path)
        return FCRNDepthPredictor(model)



class DepthPrediction:
    def __init__(self, depth_pred):
        pass