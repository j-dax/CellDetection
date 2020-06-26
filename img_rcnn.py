import sys
import random
import math
import numpy as np
import skimage.io
import skimage.color
import matplotlib
import matplotlib.pyplot as plt
from os import path, getcwd

from matplotlib.patches import Polygon

# sometimes matplotlib doesn't see tkinter
# used in isolation testing
import tkinter
matplotlib.pyplot.switch_backend('tkAgg')

import tensorflow as tf
import cv2
from mrcnn import utils
from mrcnn import visualize
from mrcnn.utils import resize_mask, resize_image
from mrcnn.model import MaskRCNN

from mask.nucleus_config import NucleusInferenceConfig
from skimage.transform import resize


class InferenceConfig(NucleusInferenceConfig):
    '''
        Config classes set many parameters used by the mrcnn library
    '''
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NAME = "Leica"


class ImageRCNN:
    '''

    '''
    def __init__(self, networkpath, modelpath):
        self._config = InferenceConfig()
        # config.display()
        self._model = MaskRCNN(mode="inference", model_dir=modelpath, config=self._config)

        # Load trained weights 
        self._model.load_weights(networkpath, by_name=True)

    def write_summary(self, filename):
        with open(filename, "w") as f:
            stdout = sys.stdout
            sys.stdout = f
            self._model.keras_model.summary()
            sys.stdout = stdout

    def get_model_inputs(self, loaded_images: list):
        # images are recast to standardized shapes
        molded_images, image_metas, windows = self._model.mold_inputs(loaded_images)
        # at this point, molded images are reshaped and ready to use in the neural net
        image_shape = molded_images[0].shape
        anchors = self._model.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (len(loaded_images),) + anchors.shape)
        
        return [molded_images, image_metas, anchors, windows]

    def unmold_predictions(self, detections, mrcnn_mask, original_image_shape, image_shape, window):
        '''
            copied from mrcnn, changes in loop allows return of probability map
        '''
        # Detections array is padded with zeros. Find the first class_id == 0.
        zero_ix = np.where(detections[:, 4] == 0)[0]
        N = zero_ix[0] if zero_ix.shape[0] > 0 else detections.shape[0]
        # Extract boxes, class_ids, scores, and class-specific masks
        boxes = detections[:N, :4]
        class_ids = detections[:N, 4].astype(np.int32)
        scores = detections[:N, 5]
        masks = mrcnn_mask[np.arange(N), :, :, class_ids]

        # Translate normalized coordinates in the resized image to pixel
        # coordinates in the original image before resizing
        window = utils.norm_boxes(window, image_shape[:2])
        wy1, wx1, wy2, wx2 = window
        shift = np.array([wy1, wx1, wy1, wx1])
        wh = wy2 - wy1  # window height
        ww = wx2 - wx1  # window width
        scale = np.array([wh, ww, wh, ww])
        # Convert boxes to normalized coordinates on the window
        boxes = np.divide(boxes - shift, scale)
        # Convert boxes to pixel coordinates on the original image
        boxes = utils.denorm_boxes(boxes, original_image_shape[:2])

        # Filter out detections with zero area. Happens in early training when
        # network weights are still random
        exclude_ix = np.where(
            (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) <= 0)[0]
        if exclude_ix.shape[0] > 0:
            boxes = np.delete(boxes, exclude_ix, axis=0)
            class_ids = np.delete(class_ids, exclude_ix, axis=0)
            scores = np.delete(scores, exclude_ix, axis=0)
            masks = np.delete(masks, exclude_ix, axis=0)
            N = class_ids.shape[0]
        # Resize masks to original image size and set boundary threshold.
        full_masks = []
        full_pmap = []
        for i in range(N):
            # Convert neural network mask to full size mask
            threshold = 0.5
            y1, x1, y2, x2 = boxes[i]
            full_pmap.append(
                skimage.transform.resize(masks[i], (y2 - y1, x2 - x1), order=1, mode="constant")
            )
            mask = np.where(full_pmap[i] >= threshold, 1, 0).astype(np.bool)

            # Put the mask in the right location.
            full_mask = np.zeros(image_shape[:2], dtype=np.bool)
            full_mask[y1:y2, x1:x2] = mask
            # full_mask = utils.unmold_mask(masks[i], boxes[i], original_image_shape)
            full_masks.append(full_mask)
        full_masks = np.stack(full_masks, axis=-1)\
            if full_masks else np.empty(original_image_shape[:2] + (0,))
        return boxes, class_ids, scores, full_masks, full_pmap

    def get_results(self, loaded_images):
        '''
            mold image, process molds in neural net,
            unmold and return outputs
        '''
        molded_images, image_metas, windows = self._model.mold_inputs(loaded_images)
        image_shape = molded_images[0].shape
        
        anchors = self._model.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (self._model.config.BATCH_SIZE,) + anchors.shape)

        detections, _, _, mrcnn_mask, _, _, _ =\
            self._model.keras_model.predict([molded_images, image_metas, anchors], verbose=0)

        results = []
        for i, image in enumerate(loaded_images):
            final_rois, final_class_ids, final_scores, final_masks, final_pmaps =\
                self.unmold_predictions(detections[i], mrcnn_mask[i],
                                        image.shape, molded_images[i].shape,
                                        windows[i])
            results.append({
                "rois": final_rois,
                "class_ids": final_class_ids,
                "scores": final_scores,
                "masks": final_masks,
                "pmaps": final_pmaps
            })
        return results


def load_image(imagepath):
    """Load the specified image and return a [H,W,3] Numpy array.
    """
    # Load image
    image = skimage.io.imread(imagepath)
    # If grayscale. Convert to RGB for consistency.
    if image.ndim != 3:
        image = skimage.color.gray2rgb(image)
    # If has an alpha channel, remove it for consistency
    if image.shape[-1] > 3:
        image = image[..., :3]
    return image


def initialize_network(networkpath, modelpath):
    '''
        init neural network, ensure h5 file is available
    '''
    if not path.exists(networkpath):
        utils.download_trained_weights(networkpath)
    img_rcnn = ImageRCNN(networkpath, modelpath)
    return img_rcnn


def detect():
    '''
        example of obtaining a single pmap and rendering it with matplotlib
    '''
    IMAGE_PATH = path.join(getcwd(), "static/img.jpg")
    network = initialize_network(path.join(getcwd(), "nucleus.h5"), path.join(getcwd(), "logs"))
    image = load_image(IMAGE_PATH)
    results = network.get_results([image])
    pmaps = results[0]["pmaps"]
    plt.imshow(pmaps[1])
    plt.show()
    return pmaps


if __name__ == "__main__":
    detect()