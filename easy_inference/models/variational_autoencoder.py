from tensorflow.contrib import keras
from typing import List

import cv2
import tensorflow
import numpy as np

from easy_inference.models import BaseModel

"""
This code is for easily running the variational autoencoder models that I make
"""


class VariationalEncoder(BaseModel):
    def __init__(self, keras_model):
        self.model = keras_model
        self.input_shape = self.model.layers[0].input_shape[1:]  # (h, w, c)

    @staticmethod
    def from_path(model_path):
        return VariationalEncoder(keras.models.load_model(model_path))

    def predict(self, imgs_bgr: List[np.ndarray]) -> List[np.ndarray]:
        preprocessed = [cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
                        for img in imgs_bgr]
        preprocessed = np.asarray(preprocessed, dtype=np.float32)
        preprocessed /= 255.

        pred = self.model.predict(preprocessed)
        return pred


class VariationalDecoder(BaseModel):
    def __init__(self, keras_model):
        self.model = keras_model
        self.output_shape = self.model.layers[-1].output_shape[1:]  # (h, w, c)

    @staticmethod
    def from_path(model_path):
        return VariationalDecoder(keras.models.load_model(model_path))

    def predict(self, latent_spaces: List[np.ndarray]) -> List[np.ndarray]:
        """
        :param latent_spaces: A list of latent vectors from the Encoder
        :return: An image in the same format as a cv2 image
        """
        preprocessed = np.asarray(latent_spaces)
        decoded = self.model.predict(preprocessed)

        # Convert to cv2 format
        postprocessed = decoded * 255
        postprocessed = postprocessed.astype(dtype=np.uint8)

        return postprocessed