from argparse import ArgumentParser
from pathlib import Path

import cv2

from easyinference import image_utils
from easyinference.depth_prediction import DepthPredictor

if __name__ == "__main__":
    parser = ArgumentParser(description="This is an example how how to use the variational autoencoder.")
    parser.add_argument("-m", "--model_path", type=str, required=True,
                        help="The path to the depth prediction model")
    parser.add_argument("-i", "--images", type=str, required=True,
                        help="The directory of images to run through the encoder+decoder one at a time")
    args = parser.parse_args()


    brain = DepthPredictor.from_path(args.model_path)

    for img_path in Path(args.images).glob("*.jpg"):

        # Load the image
        img = cv2.imread(str(img_path))

        img = image_utils.resize_and_crop(img, 512, 256)

        # Predict depth
        depth = brain.predict([img])[0]
        print(depth.shape)

        # Show the original and the depth
        cv2.imshow("INPUT", cv2.resize(img, (512, 256)))
        cv2.imshow("OUTPUT", depth)

        while cv2.waitKey(1) != ord(' '): pass