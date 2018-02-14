import tensorflow
from tensorflow.contrib import keras

import cv2
import numpy as np

"""
This code is for easily running the variational autoencoder models that I make
"""


class VariationalEncoder:
    def __init__(self, keras_model):
        self.model = keras_model
        self.input_shape = self.model.layers[0].input_shape[1:]

    @staticmethod
    def from_path(model_path):
        return VariationalEncoder(keras.models.load_model(model_path))

    def predict(self, imgs_bgr):
        processed = [cv2.resize(img, (self.input_shape[1], self.input_shape[0]))
                     for img in imgs_bgr]
        processed = np.asarray(processed)

        pred = self.model.predict(processed)
        return pred


class VariationalDecoder:
    def __init__(self, keras_model):
        self.model = keras_model

    @staticmethod
    def from_path(model_path):
        return VariationalDecoder(keras.models.load_model(model_path))

    def predict(self, latent_space):
        """
        :param latent_space: The latent vector output from the Encoder
        :return: An image in the same format as a cv2 image
        """
        pass