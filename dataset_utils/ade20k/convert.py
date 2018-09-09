from pathlib import Path
import colorsys

from dataset import Dataset
from classes import *
from img_utils import resize_and_crop

import cv2
import numpy as np


def generate_distinct_colors(n):
    """ Generate n visually distinct colors """

    HSV_tuples = [(x * 1.0 / n, 0.5, 0.5) for x in range(n)]
    rgb_out = []
    for rgb in HSV_tuples:
        rgb = list(map(lambda x: int(x*255), colorsys.hsv_to_rgb(*rgb)))
        rgb_out.append(rgb)
    return rgb_out


def is_valid(seg_img, must_include):
    """
    Decides if a DataObject has all of the necessary labels to be included in the training set
    """
    if must_include is None: return True

    # Get the "RG" channels, and the b channel of the image
    rg = seg_img[:, :, -1:0:-1]

    for cls in must_include:
        where_is_class = np.where((rg == cls.rg_code).all(axis=2))

        # If there are no pixels of that class in the image, it's not valid
        if len(where_is_class[0]) == 0: return False

    return True


def remove_irrelevant_classes(seg_img, include_classes):
    """ Sets pixels to black for all pixels that represent classes that are not relevant to this dataset"""
    h, w, _ = seg_img.shape

    # Get the "RG" channels, and the b channel of the image
    rg = seg_img[:, :, -1:0:-1]
    new_bgr = np.zeros((h, w, 3), dtype=np.uint8)
    colors = generate_distinct_colors(len(include_classes))


    for i, cls in enumerate(include_classes):
        where_is_class = np.where((rg == cls.rg_code).all(axis=2))
        new_bgr[where_is_class] = colors[i]

    return new_bgr


def convert(dataset, output_dir, crop_to=None, include_classes=None, must_include=None):
    """ Convert every image that has a specific class to the output directory, sorted in
    folders where one folder is the original images and the other folder is segmented images
    :param dataset: Dataset object
    :param output_dir: Where to output the converted dataset
    :param resize_and_crop: Will resize the image to the closest it can while maintaining aspect ratio, then crop
    from the center.
    :param include_classes: A list of Label objects regarding the classes that will be included in the output segs
    :param must_include: Optional parameter of the minimum class that must exist in the image for it to be included.
        same format as include_classes
    """

    # Create the necessary directories for outputs
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_dir = Path(output_dir).resolve()
    img_dir = Path(output_dir).joinpath("original_images")
    seg_dir = Path(output_dir).joinpath("segment_images")
    img_dir.mkdir(parents=True, exist_ok=True)
    seg_dir.mkdir(parents=True, exist_ok=True)

    for data in dataset:
        img_file = str(img_dir.joinpath(str(data.file_id) + ".png"))
        seg_file = str(seg_dir.joinpath(str(data.file_id) + ".png"))



        # Preprocess the images that are to be saved
        seg_img = data.segment
        img = data.image
        if crop_to is not None:
            seg_img = resize_and_crop(data.segment, crop_to, interpolation=cv2.INTER_NEAREST)
            img = resize_and_crop(data.image, crop_to)


        # If this picture does not have the classes we are looking for, exit
        if not is_valid(seg_img, must_include):
            continue

        # Remove all classes that are irrelevant to the search
        seg_img = remove_irrelevant_classes(seg_img, include_classes)

        # Write to file
        print("Converting ", data.file_id)
        cv2.imwrite(img_file, img)
        cv2.imwrite(seg_file, seg_img)


if __name__ == "__main__":
    dataset = Dataset("./images/train")
    convert(dataset=dataset,
            output_dir="./generated/train",
            crop_to=256,
            include_classes=[FLOOR, CARPET, WALL, WINDOW],
            must_include=[FLOOR])

    dataset = Dataset("./images/validation")
    convert(dataset=dataset,
            output_dir="./generated/test",
            crop_to=256,
            include_classes=[FLOOR, CARPET, WALL, WINDOW],
            must_include=[FLOOR])
