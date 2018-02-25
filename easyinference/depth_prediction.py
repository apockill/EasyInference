from pathlib import Path

import cv2
import numpy as np

import easyinference.load_utils as loading
from easyinference.image_utils import resize_and_crop
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
        print(imgs_bgr[0].shape)
        preprocessed = [preprocess(img) for img in imgs_bgr]
        print(preprocessed[0].shape)
        feed = {self.input_node: preprocessed}
        out = self.session.run(self.output_node, feed)

        out = [DepthMap(np.squeeze(d)) for d in out]
        return out

def preprocess(img_bgr):
    img = resize_and_crop(img_bgr, 512, 256)
    img = img.astype(np.float32) / 255
    return img

class DepthMap:
    def __init__(self, depth):
        self.depth = depth

    def normalized(self):
        """ Returns a 0-255 openCV ready-to-save image """
        return cv2.normalize(self.depth, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)