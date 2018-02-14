from pathlib import Path
from argparse import ArgumentParser

import cv2

from easyinference.variational_autoencoder import VariationalDecoder, VariationalEncoder

if __name__ == "__main__":
    parser = ArgumentParser(description="This is an example how how to use the variational autoencoder.")
    parser.add_argument("-e", "--encoder", type=str, required=True,
                        help="The path to the encoder model")
    parser.add_argument("-d", "--decoder", type=str, required=True,
                        help="The path to the decoder model")
    parser.add_argument("-i", "--image", type=str, required=True,
                        help="The video to run through the encoder/decoder, one frame at a time/")
    args = parser.parse_args()


    encoder = VariationalEncoder.from_path(args.encoder)
    decoder = VariationalDecoder.from_path(args.decoder)

    img = cv2.imread(args.image)
    pred = encoder.predict([img])
