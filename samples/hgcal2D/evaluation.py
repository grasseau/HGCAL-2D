import os
import sys
import random
import math
import re
import time
import numpy as np
import hgcal2D as hg

# Root directory of the project\n",
ROOT_DIR = os.path.abspath("../../")
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn import utils

import matplotlib as plt

# GG : Limits the numbet of threads used & use 
# legacy threads
os.environ["OMP_NUM_THREADS"] = "10"
os.environ["USE_SIMPLE_THREADED_LEVEL3"] = "1"

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InferenceConfig(hg.Hgcal2DConfig):
    """Configuration for Inference on the toy hgcal2D dataset.
    """
    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    # IMAGES_PER_GPU = 8
    IMAGES_PER_GPU = 1

# GG The place ?
def get_ax(rows=1, cols=1, size=8):
     """Return a Matplotlib Axes array to be used in
     all visualizations in the notebook. Provide a
     central point to control graph sizes.
     
     Change the default size attribute to control the size
     of rendered images
     """
     _, ax = plt.pyplot.subplots(rows, cols, figsize=(size*cols, size*rows))
     return ax

config = InferenceConfig()
config.display()

# Training dataset
dataset_train = hg.Hgcal2DDataset()
dataset_train.load_hgcal2D('train.obj', 100, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()
    
# Validation dataset
dataset_val = hg.Hgcal2DDataset()
dataset_val.load_hgcal2D('eval.obj', 20, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()


# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
# Trainin 100 ev, epoch=10
#model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190524T1939/mask_rcnn_hgcal2d_0010.h5")
# Trainin 200 ev, epoch=40
model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190703T0052/mask_rcnn_hgcal2d_0040.h5")
####################################################"
# Trainin 200 ev, epoch=40, obj 2-8 (1)
# model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190828T1450/mask_rcnn_hgcal2d_0040.h5")
# Idem + anchors parameters (2)
# model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190829T1324/mask_rcnn_hgcal2d_0040.h5")
# Idem + anchors parameters (3)
# model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190829T1956/mask_rcnn_hgcal2d_0040.h5")
# Idem + anchors parameters (4)
# model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190830T1155/mask_rcnn_hgcal2d_0040.h5")
# BACKBONE (5)
#model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190830T1613/mask_rcnn_hgcal2d_0040.h5")
# Contigous BACKBONE (6)
# model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190902T1136/mask_rcnn_hgcal2d_0040.h5")
# Train =1000, eval = 50 (8)
model_path = os.path.join(ROOT_DIR, "logs/hgcal2d20190902T2048/mask_rcnn_hgcal2d_0040.h5")








# model_path = model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

###  DETECTION  ###


# Test on a random image
for i in range(20): 
  # Preparing grid plot
  """
  ax1 = plt.subplot2grid((3, 3), (0, 0), colspan=3)
  ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
  ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
  ax4 = plt.subplot2grid((3, 3), (2, 0))
  ax5 = plt.subplot2grid((3, 3), (2, 1))
  """
  ax1 = plt.pyplot.subplot2grid((1, 2), (0, 0))
  ax2 = plt.pyplot.subplot2grid((1, 2), (0, 1))

  image_id = i
  original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
     modellib.load_image_gt(dataset_val, config, 
                            image_id, use_mini_mask=False)

  log("original_image", original_image)
  log("image_meta", image_meta)
  log("gt_class_id", gt_class_id)
  log("gt_bbox", gt_bbox)
  log("gt_mask", gt_mask)

  visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, 
                            dataset_train.class_names,                               
                            title="Original " + str(image_id), 
                            ax=ax1)
#                            figsize=(8, 8))


  results = model.detect([original_image], verbose=1)

  r = results[0]
  AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
  print("AP, precisions, recalls, overlaps : ", AP, precisions, recalls, overlaps)
  visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                             dataset_val.class_names, r['scores'],
                             title="Inference " + str(image_id), 
                              ax=ax2)
  plt.pyplot.show()

###  EVALUATION ###

# Compute VOC-Style mAP @ IoU=0.5
# Running on 10 images. Increase for better accuracy.
image_ids = range(10)
APs = []
for image_id in image_ids:
    # Load image and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP, precisions, recalls, overlaps =\
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs.append(AP)
    
print("mAP: ", np.mean(APs))


