import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import sys
import random
import math
import re
import time
import numpy as np

import model as modellib
from model import log

from train_config import BowlConfig
from train_config import init_with
from train_config import epochs
from train_config import layers_to_train
from train_config import validation_split

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


config = BowlConfig()
config.display()

## Split train and validation sets
data_ids = np.asarray(os.listdir('../data/train/'))

train_size = int((1 - validation_split) * len(data_ids))

indices = np.random.permutation(data_ids.shape[0])
training_idx, val_idx = indices[:train_size], indices[train_size:]
tr_ids, val_ids = data_ids[training_idx], data_ids[val_idx]

## Read test data indices
test_ids = np.asarray(os.listdir('../data/test/'))


# Training dataset
dataset_train = BowlDataset()
dataset_train.load_shapes('../data/train/', tr_ids, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = BowlDataset()
dataset_val.load_shapes('../data/train/', val_ids, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# Test dataset
dataset_test = BowlDataset()
dataset_test.load_shapes('../data/test/', test_ids, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_test.prepare()


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)


if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)


# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=epochs,
            layers=layers_to_train)
