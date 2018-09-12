from pathlib import Path

import ujson
import cv2


class COCODataset:
    def __init__(self, coco_dir, train=True):
        if train:
            self.ann_path = Path(
                coco_dir) / "annotations" / "instances_train2014.json"
            self.img_dir = Path(coco_dir) / "train2014"
        else:
            self.ann_path = Path(
                coco_dir) / "annotations" / "instances_val2014.json"
            self.img_dir = Path(coco_dir) / "val2014"

        label_map_path = Path(__file__).parent / "coco_label_map.json"
        self.coco_label_map = ujson.load(open(label_map_path, "r"))
        self.coco_data = ujson.load(open(self.ann_path, "r"))
        self.coco_annotations = self.coco_data["annotations"]
        self.coco_images = self.coco_data["images"]

    def __iter__(self):
        # Create map of image ID to annotation
        img_to_annotations = {img["id"]: [] for img in self.coco_images}
        for annotation in self.coco_annotations:
            img_to_annotations[annotation["image_id"]].append(annotation)

        for img_info in self.coco_images:
            annotations = img_to_annotations[img_info["id"]]
            annotation = ImageInfo(image_info_dict=img_info,
                                   img_dir=self.img_dir,
                                   annotation_dicts=annotations,
                                   label_map=self.coco_label_map)
            yield annotation


class Annotation:
    def __init__(self, ann_dict, label_map):
        self.id = ann_dict["id"]
        self.cat_id = ann_dict["category_id"]
        self.cat_name = label_map[str(self.cat_id)]

        box = ann_dict["bbox"]
        self.width = box[-2]
        self.height = box[-1]
        self.rect = [int(round(box[0], 0)),
                     int(round(box[1], 0)),
                     int(round(box[0] + box[2], 0)),
                     int(round(box[1] + box[3], 0))]

    def crop(self, frame):
        """Return the cropped portion of the frame that contains the object"""
        return frame[self.rect[1]:self.rect[3], self.rect[0]:self.rect[2]]

    def __repr__(self):
        return str(self.__dict__)


class ImageInfo:
    def __init__(self, image_info_dict, img_dir, annotation_dicts, label_map):
        self.id = image_info_dict["id"]
        self.annotations = [Annotation(ann_dict=ann_dict,
                                       label_map=label_map)
                            for ann_dict in annotation_dicts]
        self.path = Path(img_dir) / image_info_dict["file_name"]

    def filter_annotations(self, category_names=None):
        annotations = self.annotations.copy()
        if category_names is not None:
            annotations = (a for a in annotations if
                           a.cat_name in category_names)

        return list(annotations)

    def __repr__(self):
        return str(self.id) + " " + str(self.path)

    def load_frame(self):
        return cv2.imread(str(self.path))
