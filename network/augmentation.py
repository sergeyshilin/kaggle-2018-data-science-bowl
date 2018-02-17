# https://www.kaggle.com/gaborfodor/augmentation-methods

import cv2
import params
import numpy as np
import pandas as pd
from keras.preprocessing import image
from os.path import join
from skimage.transform import rotate as rt
from skimage.transform import resize
import matplotlib.pyplot as plt

np.random.seed(1987)

aug_flip_proba = params.aug_flip_proba
aug_max_angle = params.aug_max_angle
aug_max_shift = params.aug_max_shift
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

def rotate(det, order = 5, img = None, mask = None, max_angle = 30.0, det_angle = None, image = None, ini_shape = None):
    if not det:
        angle = np.random.uniform(-max_angle, max_angle)
        return resize(rt(img, angle, preserve_range = True, resize = True, order = order), \
                      img.shape[:2], order = order), \
                resize(rt(mask, angle, preserve_range = True, resize = True, order = order), \
                       mask.shape[:2], order = order)
    else:
        if image:
            det_angle = np.random.uniform(-max_angle, max_angle)
            img_rt = rt(img, det_angle, preserve_range = True, resize = True, order = order)
            return resize(img_rt, img.shape[:2], order = order), (det_angle, img_rt.shape[:2])
        else:
            mask_big = resize(mask, ini_shape, order = order)
            mask_big = rt(mask_big, det_angle, preserve_range = True, resize = True, order = order)
            dx = (mask_big.shape[0] - mask.shape[0]) // 2
            dy = (mask_big.shape[1] - mask.shape[1]) // 2
            return mask_big[dx:(dx + mask.shape[0]), dy:(dy + mask.shape[1]), ...]

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

def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        coef = np.array([[[0.114, 0.587, 0.299]]])  # rgb to gray (YCbCr)
        gray = img * coef
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img

def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img

def random_augmentation(img, mask):
    img = random_brightness(img, limit = (-0.5, 0.5), u = 0.5)
    img = random_contrast(img, limit = (-0.5, 0.5), u = 0.5)
    for transform in np.random.choice(possible_transforms, size = len(possible_transforms), replace = False):
        if 'flip' in transform:
            img, mask = flip(det = False, axis = int(transform[-1]), img = img, mask = mask, proba = aug_flip_proba)
        if 'shift' in transform:
            img, mask = shift(det = False, axis = int(transform[-1]), img = img, mask = mask, max_shift = aug_max_shift)
        if transform == 'rotate':
            img, mask = rotate(det = False, img = img, mask = mask, max_angle = aug_max_angle)
    return img, mask

def deterministic_augmentation(obj, image = True, aug_params = []):
    if image:
        obj = random_brightness(obj, limit = (-0.5, 0.5), u = 0.5)
        obj = random_contrast(obj, limit = (-0.5, 0.5), u = 0.5)
        aug_params = []
        for transform in np.random.choice(possible_transforms, size = len(possible_transforms), replace = False):
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
                obj = rotate(det = True, mask = obj, image = False, det_angle = -transform[1][0], ini_shape = transform[1][1])
    return obj, aug_params
