from argparse import ArgumentParser
from pathlib import Path

import cv2

"""
Dataset can be found here: 
https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/shape/poselets/
"""


class Dataset:
    class ImageLabel:
        def __init__(self, label_line, image_dir):
            indexes = label_line.split()
            self.name = indexes[0]
            self.path = Path(image_dir) / self.name

            self.rect = [int(round(float(indexes[1]), 0)),
                         int(round(float(indexes[2]), 0)),
                         int(round(float(indexes[1]) + float(indexes[3]), 0)),
                         int(round(float(indexes[2]) + float(indexes[4]), 0))]

            def parse_val(index):
                """Returns None, True, or False. None if the value isn't there
                """
                if int(indexes[index]) == -1:
                    return None
                return int(indexes[index]) == 1

            self.is_male = parse_val(5)
            self.has_long_hair = parse_val(6)
            self.has_glasses = parse_val(7)
            self.has_hat = parse_val(8)
            self.has_tshirt = parse_val(9)
            self.has_long_sleeves = parse_val(10)
            self.has_shorts = parse_val(11)
            self.has_jeans = parse_val(12)
            self.has_long_pants = parse_val(13)

        def get(self, class_name):
            class_map = {
                "is_male": self.is_male,
                "has_long_hair": self.has_long_hair,
                "has_glasses": self.has_glasses,
                "has_hat": self.has_hat,
                "has_tshirt": self.has_tshirt,
                "has_long_sleeves": self.has_long_sleeves,
                "has_shorts": self.has_shorts,
                "has_jeans": self.has_jeans,
                "has_long_pants": self.has_long_pants}
            assert class_name in class_map.keys(), \
                class_name + " does not exist in " + str(class_map.keys())
            return class_map[class_name]

    def __init__(self, image_dir):
        label_file = Path(image_dir) / "labels.txt"
        self.split = Path(image_dir).name

        with open(label_file, "r") as file:
            label_lines = file.readlines()
        print(label_file)
        self.labels = []
        for line in label_lines:
            try:
                label = self.ImageLabel(line, image_dir)
            except ValueError as e:
                print("Skipped image" + line.strip()
                      + ", no bounding box found!")
                continue
            self.labels.append(label)

    def __iter__(self):
        for label in self.labels:
            yield label


def crop_and_save(lbl, save_to_dir, class_name, from_split):
    output_dir = Path(save_to_dir) / class_name
    output_dir.mkdir(exist_ok=True, parents=True)
    output_path = Path(output_dir) / (Path(from_split).name + lbl.name)

    image = cv2.imread(str(lbl.path))
    cropped = image[lbl.rect[1]:lbl.rect[3],
              lbl.rect[0]:lbl.rect[2]]
    cv2.imwrite(str(output_path), cropped)


def sort_dataset(dataset, save_to_dir, sort_classes):
    for label in dataset:
        has_all = True
        for class_name in sort_classes:
            if label.get(class_name) is None:
                has_all = False
                continue

            if label.get(class_name):
                crop_and_save(label, save_to_dir, class_name, dataset.split)
            else:
                crop_and_save(label, save_to_dir,
                              "not_" + class_name, dataset.split)

        # Create "False for all" category
        if has_all:
            if all([not label.get(cls) for cls in sort_classes]):
                crop_and_save(label, save_to_dir,
                              "None_of_above", dataset.split)


def main(args):
    for img_dir in args.image_dirs:
        dataset = Dataset(img_dir)
        sort_dataset(dataset, args.output_dir, args.sort_classes)


if __name__ == "__main__":
    parser = ArgumentParser(
        description="Split images into seperate directories for quick and easy"
                    " classification purposes")
    parser.add_argument("-d", "--image-dirs", type=str, required=True,
                        nargs="+",
                        help="The directory with the images to be classified")
    parser.add_argument("-o", "--output-dir", type=str, required=True,
                        help="The output directory to sort images in")
    parser.add_argument("-s", "--sort-classes", type=str, nargs="+",
                        required=True,
                        help="A list of classes to sort by. Possible classes:"
                             "is_male, has_long_hair, has_glasses, has_hat, "
                             "has_tshirt, has_long_sleeves, has_shorts, "
                             "has_jeans, has_long_pants")
    arguments = parser.parse_args()

    main(arguments)
