from pathlib import Path
from argparse import ArgumentParser

import cv2

from easy_inference.models import VariationalDecoder, VariationalEncoder

if __name__ == "__main__":
    parser = ArgumentParser(description="This is an example how how to use the variational autoencoder.")
    parser.add_argument("-e", "--encoder", type=str, required=True,
                        help="The path to the encoder model")
    parser.add_argument("-d", "--decoder", type=str, required=True,
                        help="The path to the decoder model")
    parser.add_argument("-i", "--images", type=str, required=True,
                        help="The directory of images to run through the encoder+decoder one at a time")
    args = parser.parse_args()


    encoder = VariationalEncoder.from_path(args.encoder)
    decoder = VariationalDecoder.from_path(args.decoder)

    for img_path in Path(args.images).glob("*.png"):

        # Load the image
        img = cv2.imread(str(img_path))

        # Encode the image
        pred = encoder.predict([img])
        print(pred)
        # Decode the image
        img_p = decoder.predict(pred)

        # Show the encoded and decoded image
        cv2.imshow("INPUT", img)
        cv2.imshow("OUTPUT", img_p[0])

        while cv2.waitKey(1) != ord(' '): pass