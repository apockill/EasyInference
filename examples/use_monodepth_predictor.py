from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np

from easyinference import image_utils
from easyinference.models import MonoDepthPredictor


if __name__ == "__main__":
    parser = ArgumentParser(description="This is an example how how to use the variational autoencoder.")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="The path to the depth prediction model")
    parser.add_argument("-i", "--images", type=str, required=True,
                        help="The directory of images to run through the encoder+decoder one at a time")
    args = parser.parse_args()

    # brain = DepthPredictor.from_path(args.model_path)

    for model_path in Path("../model_files/depth-prediction-monodepth").glob("*/frozen_model.pb"):
        args.model_path = model_path
        print(model_path)
        brain = MonoDepthPredictor.from_path(args.model_path)

        for img_path in Path(args.images).glob("*.jpg"):

            # Load the image
            img = cv2.imread(str(img_path))

            # Predict depth
            pred = brain.predict(img)


            depth = pred.normalized()

            # Show the original and the depth
            cv2.imshow("INPUT", image_utils.resize_and_crop(img, 512, 256))
            cv2.imshow("OUTPUT", depth)


            # while cv2.waitKey(1) != ord(' '): pass
            cv2.waitKey(1)
            write_to = args.images + "\\" + \
                       img_path.stem + \
                       str(args.model_path).split("\\")[-2]

            cv2.imwrite(write_to + ".png", depth)
