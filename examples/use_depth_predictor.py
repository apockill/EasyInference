from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from easyinference import image_utils
from easyinference.models import MonoDepthPredictor, FCRNDepthPredictor


PREDICTORS = {"monodepth": MonoDepthPredictor, "fcrn": FCRNDepthPredictor}

if __name__ == "__main__":
    parser = ArgumentParser(description="This is an example how how to use "
                                        "the variational autoencoder.")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="The path to the depth prediction model")
    parser.add_argument("-t", "--model_type", type=str, required=True,
                        help="The type of depth predictor: " +
                             ", ".join(PREDICTORS))
    parser.add_argument("-i", "--images", type=str, required=True,
                        help="The directory of images to run through the "
                             "encoder+decoder one at a time")
    args = parser.parse_args()


    brain = PREDICTORS[args.model_type].from_path(args.model_path)

    for img_path in Path(args.images).glob("*.jpg"):

        # Load the image
        img = cv2.imread(str(img_path))

        # Predict depth
        pred = brain.predict([img])[0]
        depth = pred.normalized()

        # Show the original and the depth
        cv2.imshow("INPUT", img)
        cv2.imshow("OUTPUT", depth)

        while cv2.waitKey(1) != ord(' '): pass

