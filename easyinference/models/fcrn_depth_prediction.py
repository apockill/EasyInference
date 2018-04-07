from typing import List

import cv2
import numpy as np

from easyinference import load_utils
from easyinference.image_utils import resize_and_pad
from easyinference.models import BaseModel
from easyinference.predictions import DepthMap


class FCRNDepthPredictor(BaseModel):
    WIDTH = 304
    HEIGHT = 228

    def __init__(self, model_bytes):
        self.graph, self.session = load_utils.parse_tf_model(model_bytes)

        self.input_node = self.graph.get_tensor_by_name("Placeholder:0")
        self.output_node = self.graph.get_tensor_by_name("ConvPred/ConvPred:0")

    @staticmethod
    def from_path(model_path):
        model = load_utils.load_tf_model(model_path)
        return FCRNDepthPredictor(model)

    def predict(self, imgs_bgr: List[np.ndarray]) -> List[DepthMap]:
        """
        This model only accepts color images of resolution (304, 228)
        :param imgs_bgr: A color BGR image
        :return:
        """
        preprocessed = [_preprocess(img) for img in imgs_bgr]
        feed = {self.input_node: preprocessed}
        out = self.session.run(self.output_node, feed)
        return [DepthMap(pred) for pred in out]


def _preprocess(img_bgr):
    img_bgr = resize_and_pad(img_bgr,
                             FCRNDepthPredictor.WIDTH,
                             FCRNDepthPredictor.HEIGHT)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    return img_rgb

