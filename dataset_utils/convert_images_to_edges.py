from argparse import ArgumentParser
from pathlib import Path

import cv2
import numpy as np


def auto_canny(image, sigma=0.5):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


def pyimagesearch_canny(image, blur_kernel_size=1):
    # load the image, convert it to grayscale, and blur it slightly
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_kernel_size, blur_kernel_size), 0)

    # apply Canny edge detection using a wide threshold, tight
    # threshold, and automatically determined threshold
    wide = cv2.Canny(blurred, 10, 240)
    tight = cv2.Canny(blurred, 225, 250)
    auto = auto_canny(blurred)

    # show the images
    cv2.imshow("Original", image)
    cv2.imshow("Edges", np.hstack([wide, tight, auto]))

    return auto


def convert_image_using_contours(frame):
    canny = pyimagesearch_canny(frame)
    im2, contours, hierarchy = cv2.findContours(
        canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    # Filter out tiny contours
    image_area = frame.shape[0] * frame.shape[1]
    min_area = image_area * .0005

    # contours.sort(key=cv2.contourArea, reverse=True)
    # contours.sort(key=lambda c: cv2.arcLength(c, True), reverse=True)
    def solidity(cnt):
        area = cv2.contourArea(cnt)
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area==0:
            return 0
        solidity = float(area) / hull_area
        print("AYY", solidity)
        return solidity

    contours.sort(key=cv2.contourArea, reverse=True)
    contours = contours[:50]
    # contours.sort(key=solidity, reverse=True)
    # contours = contours[:50]


    # draw in blue the contours that were founded
    contoured = np.zeros(canny.shape, dtype=np.uint8)
    contoured = cv2.drawContours(contoured, contours, -1, 255, 1)

    # Invert the contours so that the background is white
    contoured = 255 - contoured
    return contoured


def convert_dir(input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=False)
    assert input_dir.is_dir()

    image_paths = list(input_dir.glob("*.jpg")) + \
                  list(input_dir.glob("*.png"))
    for image_path in image_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            print("Error: Could not read", image_path)
            continue

        edged = convert_image_using_contours(frame)
        save_to = output_dir / image_path.name
        cv2.imwrite(str(save_to), edged)

        cv2.imshow('ayy', edged)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Convert a directory of images to edges")
    parser.add_argument("--input_dir", required=True,
                        help="Path to the directory with input images")
    parser.add_argument("--output_dir", required=True,
                        help="Path to save output images in")
    args = parser.parse_args()
    convert_dir(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir))
