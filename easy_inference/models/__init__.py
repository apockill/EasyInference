from .base_class import BaseModel, TensorflowBaseModel

from .deeplab_segmentation import DeeplabImageSegmenter
from .fcrn_depth import FCRNDepthPredictor
from .monodepth import MonoDepthPredictor

from .object_detection import ObjectDetector
from .star_gan import StarGanGenerator
from .slim_image_classification import ImageClassifier
from easy_inference.labels import Classification

# Try Keras related imports
try:
    from .variational_autoencoder import VariationalDecoder, VariationalEncoder
except ModuleNotFoundError:
    import logging
    logging.warning("Some imports failed because keras is not installed")