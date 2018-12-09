from typing import List

import numpy as np
import cv2

from .labels import Detection

"""Some helpful drawing functions for use with openCV"""


def draw_detection(image: np.ndarray, detection: Detection, border=2):
    """ Draw a detection onto a cv2 frame"""

    cv2.rectangle(image,
                  pt1=(int(detection.rect[0]), int(detection.rect[1])),
                  pt2=(int(detection.rect[2]), int(detection.rect[3])),
                  color=detection.display_color,
                  thickness=border)

    # Create the text for the detection name
    label_text = detection.name + " (" + \
                 str(round(detection.confidence * 100, 0)) + "%)"

    # Create the bounding box that will surround the detection
    text_size = cv2.getTextSize(text=label_text,
                                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                fontScale=0.3,
                                thickness=1)[0]
    label_rect = [int(detection.rect[0]),
                  int(detection.rect[1]) - text_size[1] + 7,
                  int(detection.rect[0]) + text_size[0] + 7,
                  int(detection.rect[1])]

    # Create a colored box that will have the detection name in it
    cv2.rectangle(image,
                  tuple(label_rect[:2]), tuple(label_rect[2:]),
                  tuple(detection.display_color), -1)

    # Write the class name of the detection onto the bounding box
    font_center = (label_rect[0] + 5,
                   label_rect[1] + int(text_size[1] + 10) - 5)
    cv2.putText(image, label_text, tuple(font_center),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3,
                (0, 0, 0), thickness=1)
