# https://www.kaggle.com/gaborfodor/augmentation-methods

import cv2
import params
import numpy as np
import pandas as pd
from keras.preprocessing import image
from os.path import join
from skimage.transform import rotate as rt
import matplotlib.pyplot as plt

np.random.seed(1987)

aug_flip_proba = params.aug_flip_proba
aug_max_angle = params.aug_max_angle
aug_max_shift = params.aug_max_shift
aug_num_transforms = params.aug_num_transforms
possible_transforms = params.aug_possible_transforms

def flip(det, axis, img = None, mask = None, proba = 0.5, det_param = None, image = None):
    if not det:
        if np.random.random() < proba:
            img = np.flip(img, axis = axis)
            mask = np.flip(mask, axis = axis)
        return img, mask
    else:
        if image:
            det_param = (np.random.random() < proba)
            if det_param:
                img = np.flip(img, axis = axis)
            return img, det_param
        else:
            if det_param:
                mask = np.flip(mask, axis = axis)
            return mask

def rotate(det, img = None, mask = None, max_angle = 30.0, det_angle = None, image = None):
    if not det:
        angle = np.random.uniform(-max_angle, max_angle)
        return rt(img, angle, preserve_range = True), rt(mask, angle, preserve_range = True)
    else:
        if image:
            det_angle = np.random.uniform(-max_angle, max_angle)
            return rt(img, det_angle, preserve_range = True), det_angle
        else:
            return rt(mask, det_angle, preserve_range = True)

def shift(det, axis, img = None, mask = None, max_shift = 0.3, det_shift = None, image = None):
    if not det:
        shift = int(np.random.uniform(-max_shift, max_shift) * img.shape[axis])
        return np.roll(img, shift, axis = axis), np.roll(mask, shift, axis = axis)
    else:
        if image:
            det_shift = int(np.random.uniform(-max_shift, max_shift) * img.shape[axis])
            return np.roll(img, det_shift, axis = axis), det_shift
        else:
            return np.roll(mask, det_shift, axis = axis)

def random_augmentation(img, mask):
    for _ in range(aug_num_transforms):
        transform = np.random.choice(possible_transforms)
        if 'flip' in transform:
            img, mask = flip(det = False, axis = int(transform[-1]), img = img, mask = mask, proba = aug_flip_proba)
        if 'shift' in transform:
            img, mask = shift(det = False, axis = int(transform[-1]), img = img, mask = mask, max_shift = aug_max_shift)
        if transform == 'rotate':
            img, mask = rotate(det = False, img = img, mask = mask, max_angle = aug_max_angle)
    return img, mask

def deterministic_augmentation(obj, image = True, aug_params = []):
    if image:
        aug_params = []
        for _ in range(aug_num_transforms):
            transform = np.random.choice(possible_transforms)
            if 'flip' in transform:
                obj, flip_p = flip(det = True, axis = int(transform[-1]), img = obj, proba = aug_flip_proba, image = True)
                aug_params.append([transform, flip_p])
            if 'shift' in transform:
                obj, shift_p = shift(det = True, axis = int(transform[-1]), img = obj, max_shift = aug_max_shift, image = True)
                aug_params.append([transform, shift_p])
            if transform == 'rotate':
                obj, rotate_p = rotate(det = True, img = obj, max_angle = aug_max_angle, image = True)
                aug_params.append([transform, rotate_p])
    else:
        for transform in reversed(aug_params):
            if 'flip' in transform[0]:
                obj = flip(det = True, axis = int(transform[0][-1]), mask = obj, image = False, det_param = transform[1])
            if 'shift' in transform[0]:
                obj = shift(det = True, axis = int(transform[0][-1]), mask = obj, image = False, det_shift = -transform[1])
            if 'rotate' in transform[0]:
                obj = rotate(det = True, mask = obj, image = False, det_angle = -transform[1])
    return obj, aug_params
