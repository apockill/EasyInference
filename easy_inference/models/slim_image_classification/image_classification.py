from typing import List
import json

import numpy as np

import easy_inference.load_utils as loading
from easy_inference.models.base import TensorflowBaseModel
from easy_inference.predictions import Classification
from .preprocessing import CLASSIFIERS, ClassifierParams

"""
This class is able to run models from the tensorflow models/research/slim
repository, findable here: 
https://github.com/tensorflow/models/tree/master/research/slim
"""


class ImageClassifier(TensorflowBaseModel):
    def __init__(self, model_bytes, labels, model_name):
        assert model_name in CLASSIFIERS, \
            "{model_name} is not a supported model! Supported models:" + \
            str(CLASSIFIERS.keys())

        self.model_params: ClassifierParams = CLASSIFIERS[model_name]

        self.label_map = labels
        self.graph, self.session = loading.parse_tf_model(model_bytes)

        self.input_tensor = self.graph.get_tensor_by_name(
            self.model_params.input_node)
        self.output_tensor = self.graph.get_tensor_by_name(
            self.model_params.output_node)

    def predict(self, imgs_bgr: List[np.ndarray]) -> List[Classification]:
        # Prepare the images for being run by the model
        preprocessed = [self.model_params.preprocess(img=img.copy(),
                                                     config=self.model_params)
                        for img in imgs_bgr]

        # Batch predict
        feed = {self.input_tensor: preprocessed}
        outs = self.session.run(self.output_tensor, feed_dict=feed)

        # Create the classification predictions
        preds = []  # [pred for image 1, pred for image 2]
        for out in outs:
            class_id = np.argmax(out)
            winning_name = self.label_map[int(class_id)]

            scores = {self.label_map[i]: out[i]
                      for i in range(len(self.label_map))}
            assert scores[winning_name] == out[class_id]  # TODO: Remove line

            pred = Classification(winning_name, scores)
            preds.append(pred)
        return preds
