from argparse import ArgumentParser
from pathlib import Path
import shutil

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


class Image:
    def __init__(self, image_path, labels, class_names):
        """
        :param filename: The name of the image file
        :param labels: [-1, 1, 1, -1, ... ]
        """
        self.image_path = image_path

        self.labels = {}
        for i, class_name in enumerate(class_names):
            if labels[i] == 1:
                self.labels[class_name] = True
            elif labels[i] == -1:
                self.labels[class_name] = False
            else:
                raise ValueError("Something was mislabeled!")

    def __repr__(self):
        return "File " + self.filename + \
               " Labels " + ", ".join(map(str, self.labels))

    def copy_to(self, new_dir: Path):
        shutil.copyfile(self.image_path, new_dir / self.image_path.name)


class CelebA:
    def __init__(self, images_dir: Path, attr_file: Path):
        self.images_dir = images_dir

        with open(str(attr_file), "r") as file:
            lines = file.readlines()

        # Pop the first line (length of dataset)
        lines.pop(0)
        class_names = lines.pop(0).split()
        labels = [line.split() for line in lines]

        self.images = [Image(images_dir / line[0],
                             list(map(int, line[1:])),
                             class_names)
                       for line in labels]

    def copy_images(self, with_attrs, to_dir, save_others=False):
        """Copy images with specific attributes to a new directory.
        A directory for each attribute will be created.
        :param with_attrs: ["attr name 1", "attr name 2", etc]
        """
        to_dir = Path(to_dir)
        to_dir.mkdir(parents=True, exist_ok=True)

        for image in self.images:
            for attr in with_attrs:
                if image.labels[attr]:
                    save_to = to_dir / attr
                    save_to.mkdir(parents=True, exist_ok=True)
                    image.copy_to(save_to)
                elif save_others:
                    save_to = to_dir / "Other"
                    save_to.mkdir(parents=True, exist_ok=True)
                    image.copy_to(save_to)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--images_dir", required=True,
                        help="Path to the images directory")
    parser.add_argument("--attr_file", required=True,
                        help="Path to the list_attr_celeba.txt file")
    parser.add_argument("--copy_to", required=True,
                        help="Path to copy new (sorted) images")
    parser.add_argument("--attrs", nargs="+", required=True, type=str,
                        help="A list of attributes you would like sorted."
                             " Attr options: " + ", ".join(ATTRIBUTES))
    parser.add_argument("--save_others", action="store_true",
                        help="With this flag, any image without the desired"
                             " attributes will be saved into an 'other' folder")
    args = parser.parse_args()

    dataset = CelebA(images_dir=Path(args.images_dir),
                     attr_file=Path(args.attr_file))
    dataset.copy_images(args.attrs, args.copy_to, save_others=args.save_others)