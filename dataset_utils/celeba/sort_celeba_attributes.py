from argparse import ArgumentParser
from pathlib import Path

from dataset_utils.celeba.celeba import CelebA, ATTRIBUTES

def copy_images(dataset, with_attrs, to_dir, save_others=False):
    """Copy images with specific attributes to a new directory.
    A directory for each attribute will be created.
    :param with_attrs: ["attr name 1", "attr name 2", etc]
    """
    to_dir = Path(to_dir)
    to_dir.mkdir(parents=True, exist_ok=True)

    for image in dataset:
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
    parser.add_argument("--attr_dir", required=True,
                        help="Path to the directory containing the "
                             "list_attr_celeba.txt file")
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
                     attr_dir=Path(args.attr_dir))
    copy_images(dataset, args.attrs, args.copy_to, save_others=args.save_others)
