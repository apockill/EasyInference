import cv2
import numpy as np


class DepthMap:
    def __init__(self, depth):
        self.depth = depth

    def normalized(self):
        """ Returns a 0-255 openCV ready-to-save image """
        return cv2.normalize(self.depth, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)

    def resized(self, new_width, new_height):
        """
        This resizes the depthmap while preserving average, min, and max values
        """
        return cv2.resize(self.depth, (new_width, new_height))