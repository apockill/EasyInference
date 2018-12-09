import json
from typing import List

import cv2
import numpy as np

import easy_inference.model_loading as loading
from easy_inference.models import TensorflowBaseModel
from easy_inference.labels import Detection


class ObjectDetector(TensorflowBaseModel):
    def __init__(self, model_bytes, labels, confidence_thresh=0.5):
        self.confidence_thresh = confidence_thresh

        self.label_map = labels
        graph, self.session = loading.parse_tf_model(model_bytes)

        # Get relevant nodes from the graph
        self.input_tensor = graph.get_tensor_by_name('image_tensor:0')
        self.boxes_tensor = graph.get_tensor_by_name('detection_boxes:0')
        self.scores_tensor = graph.get_tensor_by_name('detection_scores:0')
        self.classes_tensor = graph.get_tensor_by_name('detection_classes:0')

    def predict(self, imgs_bgr: List[np.ndarray]) -> List[List['Detection']]:
        # Preprocess all of the images
        imgs_rgb = [cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for img in imgs_bgr]

        # Run inference on all of the images
        get_outputs = [self.boxes_tensor, self.scores_tensor,
                       self.classes_tensor]
        feed = {self.input_tensor: imgs_rgb}
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

                # Convert the box to rect format
                box = all_boxes[image_index][detection_index]
                rect = np.array([box[1], box[0], box[3], box[2]])
                rect = rect * [w, h, w, h]
                rect = np.round(rect, 0).astype(int)

                class_id = int(all_classes[image_index][detection_index])
                name = self.label_map[str(class_id)]
                detections.append(Detection(name, rect, score))
            all_detections.append(detections)
        return all_detections

    def close(self):
        self.session.close()

