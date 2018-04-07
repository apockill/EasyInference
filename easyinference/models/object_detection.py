import json
from typing import List

import cv2
import numpy as np

import easyinference.load_utils as loading
from easyinference.models import BaseModel


class ObjectDetector(BaseModel):
    def __init__(self, model_bytes, labels_unparsed, confidence_thresh=0.5):
        self.confidence_thresh = confidence_thresh

        self.label_map = json.loads(labels_unparsed)
        self.graph, self.session = loading.parse_tf_model(model_bytes)

        # Get relevant nodes from the graph
        self.input_node = self.graph.get_tensor_by_name('image_tensor:0')
        self.boxes_node = self.graph.get_tensor_by_name('detection_boxes:0')
        self.scores_node = self.graph.get_tensor_by_name('detection_scores:0')
        self.classes_node = self.graph.get_tensor_by_name('detection_classes:0')

    @staticmethod
    def from_path(model_path, labels_path):
        """
        :param model_path: Path to the frozen detection model *.pb
        :param labels_path: Path to the labels JSON with the following format:
        {"1": "Person", "2": "Dog", "3": "Cat"}
        :return:
        """
        model_bytes = loading.load_tf_model(model_path)
        with open(labels_path, 'r') as f:
            label_str = f.read()
        return ObjectDetector(model_bytes, label_str)

    def predict(self, imgs_bgr: List[np.ndarray]) -> List[List['Detection']]:
        # Preprocess all of the images
        imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_bgr]

        # Run inference on all of the images
        get_outputs = [self.boxes_node, self.scores_node, self.classes_node]
        feed = {self.input_node: imgs_rgb}
        all_outputs = self.session.run(get_outputs, feed_dict=feed)
        all_boxes, all_scores, all_classes = all_outputs

        all_detections = []
        for image_index in range(len(imgs_bgr)):
            h, w = imgs_bgr[image_index].shape[:2]
            detections = []
            for detection_index in range(len(all_boxes[image_index])):
                # Threshold out low-confidence detections
                score = all_scores[image_index][detection_index]
                if score < self.confidence_thresh: continue

                box = all_boxes[image_index][detection_index]
                # box = box * [w, h, w, h]
                class_id = int(all_classes[image_index][detection_index])
                name = self.label_map[str(class_id)]
                detections.append(Detection(name, box, score))
            all_detections.append(detections)
        return all_detections


class Detection:
    def __init__(self, name, rect, confidence):
        self.name = name
        self.rect = rect
        self.confidence = confidence
