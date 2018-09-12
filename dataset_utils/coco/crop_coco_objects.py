from argparse import ArgumentParser
from pathlib import Path

import cv2

from dataset_utils.coco.dataset import COCODataset

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True,
                        help="Path to the coco dataset")
    parser.add_argument("--output_dir", "-o", required=True,
                        help="Directory to save cropped images to")
    parser.add_argument("--categories", "-c", nargs="+", required=True,
                        help="The categories to crop")
    parser.add_argument("--min_area", "-a", type=int, required=True,
                        help="The minimum pixel area for this to be included")
    args = parser.parse_args()

    dataset = COCODataset(coco_dir=args.dataset)

    for image in dataset:
        people = image.filter_annotations(category_names=args.categories)

        if len(people) == 0:
            continue

        frame = image.load_frame()
        for person in people:
            area = int(person.width * person.height)
            if area < args.min_area:
                continue
            crop = person.crop(frame)
            save_to = Path(args.output_dir) / \
                      (str(image.id) + "_" + str(person.id) + ".jpg")

            cv2.imwrite(str(save_to), crop)
