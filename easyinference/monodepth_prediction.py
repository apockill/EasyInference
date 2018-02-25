from pathlib import Path

import cv2
import numpy as np

import easyinference.load_utils as loading
from easyinference.image_utils import resize_and_crop
"""
This code is for running the monodepth model. 
Code can be found here: https://github.com/mrharicot/monodepth 

All credit goes to them. This is simply a wrapper around the trained model.
"""


class MonoDepthPredictor:

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
        return MonoDepthPredictor(model_bytes)

    def predict(self, img_bgr):
        preprocessed = preprocess(img_bgr)

        feed = {self.input_node: preprocessed}
        pred_pair = self.session.run(self.output_node, feed)

        out = postprocess(pred_pair.squeeze()).astype(np.float32)
        out = DepthMap(out)
        return out


def preprocess(img_bgr):
    """ Returns two images. A left disparity and a right disparity.
    Normalized from 0 to 1. Cropped to fit the networks input size. """
    img = resize_and_crop(img_bgr, 512, 256)
    img = img.astype(np.float32) / 255
    flipped = np.fliplr(img)
    imgs = np.stack((img, flipped), 0)
    return imgs


def postprocess(disp):
    """
    This is post processing taken directly from the paper
    :param disp: The disparity predicted by the model
    :return: The postprocessed disparity
    """
    _, h, w = disp.shape
    l_disp = disp[0, :, :]
    r_disp = np.fliplr(disp[1, :, :])
    m_disp = 0.5 * (l_disp + r_disp)
    l, _ = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h))
    l_mask = 1.0 - np.clip(20 * (l - 0.05), 0, 1)
    r_mask = np.fliplr(l_mask)
    return r_mask * l_disp + l_mask * r_disp + (
                                               1.0 - l_mask - r_mask) * m_disp

class DepthMap:
    def __init__(self, depth):
        self.depth = depth

    def normalized(self):
        """ Returns a 0-255 openCV ready-to-save image """
        return cv2.normalize(self.depth, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)
