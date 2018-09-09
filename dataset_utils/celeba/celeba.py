from pathlib import Path
import shutil

import cv2


class Image:
    def __init__(self, image_path, attr_labels, class_names, face_rect):
        """
        :param filename: The name of the image file
        :param attr_labels: [-1, 1, 1, -1, ... ]
        """
        self.image_path = image_path
        self.rect = face_rect

        # Parse the attributes
        self.labels = {}
        for i, class_name in enumerate(class_names):
            if attr_labels[i] == 1:
                self.labels[class_name] = True
            elif attr_labels[i] == -1:
                self.labels[class_name] = False
            else:
                raise ValueError("Something was mislabeled!")

    def __repr__(self):
        return "File " + self.filename + \
               " Labels " + ", ".join(map(str, self.labels))

    @property
    def frame(self):
        """Load the image as a numpy array"""
        print("Loading", self.image_path)
        return cv2.imread(str(self.image_path))

    def copy_to(self, new_dir: Path):
        shutil.copyfile(self.image_path, new_dir / self.image_path.name)


class CelebA:
    def __init__(self, images_dir: Path, attr_dir: Path):
        self.images_dir = images_dir

        attr_file = Path(attr_dir) / "list_attr_celeba.txt"
        bbox_file = Path(attr_dir) / "list_bbox_celeba.txt"

        attr_labels, names, self.class_names = self._parse_attr_file(attr_file)
        rects = self._parse_bbox_file(bbox_file)

        self.images = []

        for label, img_name, rect in zip(attr_labels, names, rects):
            img = Image(image_path=Path(images_dir) / img_name,
                        attr_labels=label,
                        class_names=self.class_names,
                        face_rect=rect)
            self.images.append(img)

    def _parse_attr_file(self, attr_file):
        # Parse the attributes file
        with open(str(attr_file), "r") as file:
            lines = file.readlines()

        # Pop the first line (length of dataset)
        lines.pop(0)
        class_names = lines.pop(0).split()
        labels = [line.split() for line in lines]
        img_paths = [line[0] for line in labels]
        labels = [list(map(int, line[1:])) for line in labels]
        return labels, img_paths, class_names

    def _parse_bbox_file(self, bbox_file):
        with open(str(bbox_file), "r") as file:
            lines = file.readlines()

        # Get rid of the first two lines (not useful)
        lines = lines[2:]
        bboxes = [list(map(int, l.split()[1:])) for l in lines]
        rects = [[b[0], b[1], b[0] + b[2], b[1] + b[3]]
                 for b in bboxes]
        return rects

    def __iter__(self):
        for img in self.images:
            yield img


ATTRIBUTES = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
              'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
              'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair',
              'Bushy_Eyebrows', 'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee',
              'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones', 'Male',
              'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
              'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
              'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair',
              'Wavy_Hair', 'Wearing_Earrings', 'Wearing_Hat',
              'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie',
              'Young']
