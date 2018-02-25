from pathlib import Path

import cv2
import numpy as np

import easyinference.load_utils as loading

"""
This code is for running the DeeplabV3 model. It is using the model trained from the following repository:
https://github.com/DrSleep/tensorflow-deeplab-resnet

All credit goes to them. This is simply a wrapper around the trained model.
"""


class DepthPredictor:

    def __init__(self, model_bytes):
        # Parse everything
        self.graph, self.session = loading.parse_tf_model(model_bytes)

        self.input_node = self.graph.get_tensor_by_name("split:0")
        self.output_node = self.graph.get_tensor_by_name("model/disparities/ExpandDims:0")

    @staticmethod
    def from_path(model_path):
        """
        :param model_path: A frozen tensorflow graph, *.pb
        """
        model_path = Path(model_path).resolve()
        model_bytes = loading.load_tf_model(str(model_path))
        return DepthPredictor(model_bytes)

    def predict(self, imgs_bgr):
        preprocessed = [cv2.resize(img, (512, 256)) for img in imgs_bgr]
        print(preprocessed[0].shape)
        feed = {self.input_node: preprocessed}
        out = self.session.run(self.output_node, feed)

        out = [np.squeeze(d) for d in out]
        return out
