from .base_class import BaseModel, TensorflowBaseModel

from .deeplab_segmentation import DeeplabImageSegmenter
from .fcrn_depth import FCRNDepthPredictor
from .monodepth import MonoDepthPredictor
from .variational_autoencoder import VariationalDecoder, VariationalEncoder
from .object_detection import ObjectDetector
from .star_gan import StarGanGenerator
from .slim_image_classification import ImageClassifier
from easy_inference.labels import Classification