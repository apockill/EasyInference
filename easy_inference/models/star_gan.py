from typing import List

import numpy as np
import cv2

import easy_inference.load_utils as loading
from easy_inference.models import TensorflowBaseModel


class StarGanGenerator(TensorflowBaseModel):
    """Run a StarGan model trained with the following repo:
    https://github.com/taki0112/StarGAN-Tensorflow
    """

    def __init__(self, model_bytes):
        self.graph, self.session = loading.parse_tf_model(model_bytes)
        self.cat_input = self.graph.get_tensor_by_name('category_input:0')
        self.img_input = self.graph.get_tensor_by_name(
            'FunctionBufferingResourceGetNext:0')
        self.output_tensor = self.graph.get_tensor_by_name('generator/Tanh:0')

    def predict(self, imgs_bgr: List[np.ndarray], categories: List[float]) \
            -> List[np.ndarray]:
        processed = [self.preprocess(img) for img in imgs_bgr]
        feed = {self.img_input: processed,
                self.cat_input: np.array(categories)}
        output = self.session.run(self.output_tensor, feed_dict=feed)[0]

        output = (output + 1) * 127.5
        output = output.astype(np.uint8)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        return output

    def preprocess(self, img):
        img = cv2.resize(img, (128, 128))
        img = img.astype(np.float32) / 127.5 - 1
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return img
