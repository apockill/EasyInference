import json
from pathlib import Path

import numpy as np

import easyinference.load_utils as loading
from easyinference.models import BaseModel
from easyinference.predictions import Segmentation

"""
This code is for running the DeeplabV3 model. It is using the model trained from the following repository:
https://github.com/DrSleep/tensorflow-deeplab-resnet

All credit goes to them. This is simply a wrapper around the trained model.
"""


class DeeplabImageSegmenter(BaseModel):

    def __init__(self, model_bytes, labels_unparsed):
        # Parse everything
        self.label_map = json.loads(labels_unparsed)
        self.graph, self.session = loading.parse_tf_model(model_bytes)

        self.input_node = self.graph.get_tensor_by_name("Cast:0")
        self.output_node = self.graph.get_tensor_by_name("prediction:0")

    @staticmethod
    def from_path(model_path, labels_path):
        """
        Label path structure:
        {"0":  {"label": "background",   "color": (0,0,0)}, ...}
        :param model_path: A frozen tensorflow graph, *.pb
        :param labels_path: Path to the labels.json
        :return:
        """
        model_path = Path(model_path).resolve()
        model_bytes = loading.load_tf_model(str(model_path))
        with open(labels_path, 'r') as f:
            label_str = f.read()
        return DeeplabImageSegmenter(model_bytes, label_str)

    def predict(self, img_bgr: np.ndarray) -> Segmentation:
        feed = {self.input_node: img_bgr}
        out = self.session.run(self.output_node, feed)[0]
        out = np.squeeze(out)
        return Segmentation(out, self.label_map)


