from .base import BaseModel, TensorflowBaseModel

from .deeplab_segmentation import DeeplabImageSegmenter
from .fcrn_depth_prediction import FCRNDepthPredictor
from .monodepth_prediction import MonoDepthPredictor
from .variational_autoencoder import VariationalDecoder, VariationalEncoder
from .object_detection import ObjectDetector
from .star_gan import StarGanGenerator
from .slim_image_classification import ImageClassifier
from easy_inference.predictions import Classification