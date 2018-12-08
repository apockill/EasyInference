from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from easy_inference import image_utils
from easy_inference.models import StarGanGenerator

if __name__ == "__main__":
    parser = ArgumentParser(description="This is an example how how to use "
                                        "the star-gan generator.")
    parser.add_argument("-m", "--model-path", type=str, required=True,
                        help="The path to the star-gan generator model")
    parser.add_argument("-s", "--source", required=True,
                        help="The source file or camera to draw frames from")
    args = parser.parse_args()

    # Open the brain
    brain = StarGanGenerator.from_path(args.model_path)
    img = cv2.imread("/home/alex/Pictures/MVIMG_20180912_172945.jpg")
    generated = brain.predict([img] * 16,
                  [[[[0, 0, 0, 0, 0]]] for i in range(16)])
    cv2.imshow("ayy", generated)
    cv2.waitKey(100000)
    # # Connect to camera
    # source = int(args.source) if args.source.isdigit() else args.source
    # cap = cv2.VideoCapture(source)
    # # Start inference  loop
    # while cv2.waitKey(1) != ord('1'):
    #     _, img = cap.read()
    #     if img is None: break
    #
    #     # Predict stargan
    #     img = brain.predict([img], [0, 0, 0, 0, 0, 0])
    #
    #     cv2.imshow("Window", img)
