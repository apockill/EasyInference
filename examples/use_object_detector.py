from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from easy_inference import image_utils
from easy_inference.models import ObjectDetector

if __name__ == "__main__":
    parser = ArgumentParser(description="This is an example how how to use "
                                        "the variational autoencoder.")
    parser.add_argument("-m", "--model-path", type=str, required=True,
                        help="The path to the object detection model")
    parser.add_argument("-l", "--labels-path", required=True,
                        help="Path to the labels for this model")
    parser.add_argument("-s", "--source", required=True,
                        help="The source file or camera to draw frames from")
    args = parser.parse_args()

    # Connect to camera
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)

    # Open the brain
    brain = ObjectDetector.from_path(args.model_path, args.labels_path)

    # Start inference  loop
    while cv2.waitKey(1) != ord('1'):
        _, img = cap.read()
        if img is None: break

        # Predict depth
        pred = brain.predict([img])[0]
        print(pred)

        cv2.imshow("Window", img)
