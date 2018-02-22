from pathlib import Path
import argparse

import cv2

from dataset import VOCDataset


def save_object(obj, action_types, save_dir, save_negatives=False, must_include_all_actions=True, filetype='.png'):
    """ Save the cropped object image to a directory inside of save_dir with
    the same name as the action_type that this object is exhibiting. Aka,
    if the object is 'walking', a cropped image of that person walking will
    be saved to save_dir/walking/img_id.jpg

    If the object is not exhibiting an action from action_types, it will be
    saved to save_dir/other if save_negatives is True """

    # Find out if this object has positive AND OR negative labels for each action_type
    if must_include_all_actions:
        all_types = obj.positive_actions + obj.negative_actions
        if not all([a in all_types for a in action_types]):
            return

    # Find out which action is being exhibited, if any
    category = None
    for action in obj.positive_actions:
        if action in action_types:
            category = action
            break
    category = "other" if category is None else category

    if not save_negatives and category == "other":
        return

    # Save the object to its appropriate category folder
    save_to_dir = Path(save_dir) / category
    Path(save_to_dir).mkdir(parents=True, exist_ok=True)
    save_to = save_to_dir / (str(obj.count) + Path(obj.img_path).stem + filetype)

    # Open the image, get the objects crop, and save it as an image
    img = cv2.imread(obj.img_path)
    cropped = img[obj.rect[1]:obj.rect[3],
                  obj.rect[0]:obj.rect[2]]
    print(save_to)
    cv2.imwrite(str(save_to), cropped)



def main(args):
    dataset = VOCDataset(args.images_dir, args.labels_dir)


    # Saved cropped images of each object to categorized folders
    for img in dataset:
        for obj in img.objects:
            save_object(obj,
                        args.action_types,
                        args.save_dir,
                        must_include_all_actions=False,
                        save_negatives=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=
                                     "Get all the action classification pictures,"
                                     "copy them into seperate folders with the"
                                     "appropriate action written on them.")
    parser.add_argument("--action_types", type=str, required=True, nargs="+",
                        help="List of file paths for the actions to extract")
    parser.add_argument("--save_dir", type=str, required=True,
                        help="Path to save the categorized action images")
    parser.add_argument("--images_dir", type=str, required=True,
                        help="Path of the VOC2012 images")
    parser.add_argument("--labels_dir", type=str, required=True,
                        help="Path  to the VOC2012 annotations")
    args = parser.parse_args()

    main(args)
