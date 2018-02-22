import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import easyinference.loading as loading


"""
This code is for running the DeeplabV3 model. It is using the model trained from the following repository:
https://github.com/DrSleep/tensorflow-deeplab-resnet

All credit goes to them. This is simply a wrapper around the trained model.
"""


class DepthPredictor:

    def __init__(self, model_bytes):
        # Parse everything
        self.graph, self.session = loading.parse_tf_model(model_bytes)

        # self.input_node = self.graph.get_tensor_by_name("Variable/initial_value:0")
        self.output_node = self.graph.get_tensor_by_name("model/disparities/ExpandDims:0")

    @staticmethod
    def from_path(model_path):
        """
        :param model_path: A frozen tensorflow graph, *.pb
        """
        model_path = Path(model_path).resolve()
        model_bytes = loading.load_tf_model(str(model_path))
        return DepthPredictor(model_bytes)

    def predict(self, img_bgr):
        feed = {self.input_node: img_bgr}
        out = self.session.run(self.output_node, feed)[0]
        out = np.squeeze(out)
        return Segmentation(out, self.label_map)
