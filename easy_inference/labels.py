from typing import List
import random

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


class Detection:
    def __init__(self, name, rect, confidence):
        self.name = name
        self.rect = rect
        self.confidence = confidence

    @property
    def display_color(self) -> List[int]:
        """Return a color that is unique for this detections name"""
        rand = random.Random(self.name)
        return [rand.randint(0, 255),
                rand.randint(0, 255),
                rand.randint(0, 255)]

    @property
    def width(self):
        return self.rect[2] - self.rect[0]

    @property
    def height(self):
        return self.rect[3] - self.rect[1]

    @property
    def area(self):
        """Return the area of the bounding box"""
        return self.width * self.height

    def __repr__(self):
        return str(self.__dict__)


class Classification:
    def __init__(self, class_name, scores_dict):
        """
        :param class_name: The 'winning' class name
        :param scores_dict: A dict of
            {"class_name": score, "class_name_2": score}
        """
        self.class_name = class_name
        """The highest scoring class"""

        self.scores = scores_dict
        """All of the scores"""

        self.score = self.scores[class_name]
        """The score of the 'winning' class name"""

    def __getattr__(self, item):
        if item in self.scores:
            return self.scores[item]

        super().__getattribute__(item)
