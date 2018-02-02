import numpy as np
import cv2
from scipy.signal import correlate2d, convolve2d
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler


def get_data():
    return None

def identical(data):
    return data

def correlation(data):
    corr_data = np.zeros((data.shape[0], data.shape[1], data.shape[2], 3))
    corr_data[..., 0] = np.sqrt(data[..., 0] * data[..., 1])
    corr_data[..., 1] = np.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2)
    corr_data[..., 2] = np.arctan2(data[..., 0], data[..., 1])
    return corr_data

def data_normalization(data):
    rgb_arrays = np.zeros(data.shape).astype(np.float32)
    for i, data_row in enumerate(data):
        band_1 = data_row[:,:,0]
        band_2 = data_row[:,:,1]
        band_3 = data_row[:,:,2]

        r = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        g = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        b = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        rgb = np.dstack((r, g, b))
        rgb_arrays[i] = rgb
    return np.array(rgb_arrays)

def get_best_history(history, monitor='val_loss', mode='min'):
    best_iteration = np.argmax(history[monitor]) if mode == 'max' else np.argmin(history[monitor])
    loss = history['loss'][best_iteration]
    acc = history['acc'][best_iteration]
    val_loss = history['val_loss'][best_iteration]
    val_acc = history['val_acc'][best_iteration]

    return best_iteration + 1, loss, acc, val_loss, val_acc

def resize_data(data, size):
    data_upscaled = np.zeros((data.shape[0], size[0], size[1], size[2]), dtype=data.dtype)

    for i in range(len(data)):
        data_upscaled[i] = cv2.resize(data[i], (size[0], size[1]))

    return data_upscaled[:]

