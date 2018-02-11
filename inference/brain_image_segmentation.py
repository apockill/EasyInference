import json
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

import inference.loading as loading


"""
This code is for running the DeeplabV3 model. It is using the model trained from the following repository:
https://github.com/DrSleep/tensorflow-deeplab-resnet

All credit goes to them. This is simply a wrapper around the trained model.
"""


class ImageSegmenter:
    def __init__(self, model_bytes, labels_unparsed):
        # Parse everything
        self.label_map = json.loads(labels_unparsed)
        self.graph, self.session = loading.parse_tf_model(model_bytes)

        self.input_node = self.graph.get_tensor_by_name("Cast:0")
        self.output_node = self.graph.get_tensor_by_name("prediction:0")

    @staticmethod
    def from_path(model_path, label_path):
        """
        Label path structure:
        {"0":  {"label": "background",   "color": (0,0,0)}, ...}
        :param model_path: A frozen tensorflow graph, *.pb
        :param label_path: Path to the labels.json
        :return:
        """
        model_path = Path(model_path).resolve()
        model_bytes = loading.load_tf_model(str(model_path))
        label_str = open(label_path, 'r').read()
        return ImageSegmenter(model_bytes, label_str)

    def predict(self, img_bgr):
        feed = {self.input_node: img_bgr}
        out = self.session.run(self.output_node, feed)[0]
        out = np.squeeze(out)
        return Segmentation(out, self.label_map)


    def close(self):
        self.session.close()


class Segmentation:
    """ This is the output prediction from ImageSegmenter """
    def __init__(self, segmentation, label_map):
        """
        :param segmentation: (img_height, img_width) shape array, where each number corresponds to a key in the
        label_map dictionary.
        :param label_map:
        """
        self.label_map = label_map
        self.seg = segmentation

    def colored(self):
        """Decode batch of segmentation masks.

        Args:
          mask: result of inference after taking argmax.
          num_images: number of images to decode from the batch.
          num_classes: number of classes to predict (including background).

        Returns:
          A batch with num_images RGB images of the same size as the input.
        """
        new = np.zeros((*self.seg.shape[:2], 3), dtype=np.uint8)

        for label_type in self.label_map.keys():
            label_int = int(label_type)
            replace_with = np.array(self.label_map[label_type]["color"], dtype=np.uint8)
            cells_to_replace = self.seg == label_int
            new[cells_to_replace] = replace_with

        return new