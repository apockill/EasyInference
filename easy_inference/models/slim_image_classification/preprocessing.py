from collections.__init__ import namedtuple

import cv2

from easy_inference.image_utils import resize_and_pad


class ClassifierParams:
    def __init__(self, link, input_node, output_node, img_size, preprocess):
        """
        :param link: A link to a pretrained model zip or checkpoint
        :param input_node: The name of the input tensor
        :param output_node: The name of the output tensor
        :param img_size: The (img_size, img_size) shape of input tensor
        :param preprocess: The preprocessing function to take a BGR np.uint8
        image into something that can be run by the model.
        """
        self.link = link
        self.input_node = input_node
        self.output_node = output_node
        self.img_size = img_size
        self.preprocess = preprocess


def _preprocess_inception_resnet_v2(img, config: ClassifierParams):
    img = resize_and_pad(img, new_w=config.img_size, new_h=config.img_size)

    # Normalize the image to values between 0 and 1
    img = cv2.normalize(
        img, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F, dst=None)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _preprocess_resnet_v2_50(img, config: ClassifierParams):
    img = resize_and_pad(img, new_w=config.img_size, new_h=config.img_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def _preprocess_mobilenet_v1(img, config: ClassifierParams):
    img = resize_and_pad(img, new_w=config.img_size, new_h=config.img_size)

    # Normalize the image to values between 0 and 1
    img = cv2.normalize(
        img, alpha=0, beta=1,
        norm_type=cv2.NORM_MINMAX,
        dtype=cv2.CV_32F, dst=None)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


models_repo = "http://download.tensorflow.org/models/"
mobilenet_v1_224 = ClassifierParams(
    link=models_repo + "mobilenet_v1_2018_02_22/mobilenet_v1_1.0_224.tgz",
    input_node="input:0",
    output_node="MobilenetV1/Predictions/Reshape_1:0",
    img_size=224,
    preprocess=_preprocess_mobilenet_v1)
inception_resnet_v2 = ClassifierParams(
    link=models_repo + "inception_resnet_v2_2016_08_30.tar.gz",
    input_node="input:0",
    output_node="InceptionResnetV2/Logits/Predictions:0",
    img_size=299,
    preprocess=_preprocess_inception_resnet_v2)
resnet_v2_50 = ClassifierParams(
    link=models_repo + "resnet_v2_50_2017_04_14.tar.gz",
    input_node="resnet_v2_50/Pad:0",
    output_node="resnet_v2_50/predictions/Reshape_1:0",
    img_size=230,
    preprocess=_preprocess_resnet_v2_50)

CLASSIFIERS = {"resnet_v2_50": resnet_v2_50,
               "inception_resnet_v2": inception_resnet_v2,
               "mobilenet_v1": mobilenet_v1_224}
