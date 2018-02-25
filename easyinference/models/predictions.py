import cv2
import numpy as np


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
          mask: result of easyinference after taking argmax.
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


class DepthMap:
    def __init__(self, depth):
        self.depth = depth

    def normalized(self):
        """ Returns a 0-255 openCV ready-to-save image """
        return cv2.normalize(self.depth, None, alpha=0, beta=255,
                             norm_type=cv2.NORM_MINMAX,
                             dtype=cv2.CV_8U)

    def resized(self, new_width, new_height):
        """
        This resizes the depthmap while preserving average, min, and max values
        """
        return cv2.resize(self.depth, (new_width, new_height))