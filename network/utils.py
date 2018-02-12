import os
import numpy as np
from tqdm import tqdm
import cv2
from scipy.signal import correlate2d, convolve2d
from keras.preprocessing import image
from sklearn.preprocessing import StandardScaler
from skimage.morphology import label
from skimage.transform import resize

__all__ = ['DataPreprocessing']


def get_data_train(data_path, img_size):
    train_ids = np.asarray(next(os.walk(data_path))[1])
    sizes_train = []

    X_train = np.zeros((len(train_ids), img_size, img_size, 3), dtype=np.float32)
    Y_train = np.zeros((len(train_ids), img_size, img_size, 1), dtype=np.float32)

    for i, id_ in tqdm(enumerate(train_ids), total=len(train_ids)):
        path = data_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png').astype(np.float32)
        sizes_train.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_size, img_size), mode='constant', preserve_range=True)
        X_train[i] = img

        mask = np.zeros((img_size, img_size, 1), dtype=np.float32)
        for mask_file in next(os.walk(path + '/masks/'))[2]:
            mask_ = cv2.imread(path + '/masks/' + mask_file, 0)
            mask_ = np.expand_dims(
                resize(mask_, (img_size, img_size), mode='constant', preserve_range=True),
                axis=-1
            )
            mask = np.maximum(mask, mask_)

        Y_train[i] = mask

    X_train = X_train / 255.0
    Y_train = Y_train / 255.0
    return X_train, Y_train, train_ids, sizes_train


def get_data_test(data_path, img_size):
    test_ids = np.asarray(next(os.walk(data_path))[1])
    X_test = np.zeros((len(test_ids), img_size, img_size, 3), dtype=np.float32)
    sizes_test = []

    for i, id_ in tqdm(enumerate(test_ids), total=len(test_ids)):
        path = data_path + id_
        img = cv2.imread(path + '/images/' + id_ + '.png').astype(np.float32)
        sizes_test.append([img.shape[0], img.shape[1]])
        img = resize(img, (img_size, img_size), mode='constant', preserve_range=True)
        X_test[i] = img

    X_test = X_test / 255.0
    return X_test, test_ids, sizes_test

## ========= Data Preprocessing namespace ========= ##
class DataPreprocessing:

    @staticmethod
    def identical(data):
        return data

    @staticmethod
    def correlation(data):
        corr_data = np.zeros((data.shape[0], data.shape[1], data.shape[2], 3))
        corr_data[..., 0] = np.sqrt(data[..., 0] * data[..., 1])
        corr_data[..., 1] = np.sqrt(data[..., 0] ** 2 + data[..., 1] ** 2)
        corr_data[..., 2] = np.arctan2(data[..., 0], data[..., 1])
        return corr_data

    @staticmethod
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
## ========= Data Preprocessing namespace ========= ##


def get_best_history(history, monitor='val_loss', mode='min'):
    best_iteration = np.argmax(history[monitor]) if mode == 'max' else np.argmin(history[monitor])
    loss = history['loss'][best_iteration]
    acc = history['mean_iou'][best_iteration]
    val_loss = history['val_loss'][best_iteration]
    val_acc = history['val_mean_iou'][best_iteration]

    return best_iteration + 1, loss, acc, val_loss, val_acc


def get_predictions_upsampled(predictions, original_size):
    preds_test_upsampled = []

    for i in range(len(predictions)):
        preds_test_upsampled.append(
            cv2.resize(np.squeeze(predictions[i]), (original_size[i][0], original_size[i][1]))
        )

    return preds_test_upsampled


# https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines
def run_length_encode(mask):
    pixels = mask.T.flatten()
    # We need to allow for cases where there is a '1' at either end of the sequence.
    # We do this by padding with a zero at each end when needed.
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    rle = ' '.join(str(x) for x in rle)
    return rle

# https://www.kaggle.com/stainsby/fast-tested-rle-and-input-routines
def run_length_decode(rle_str, mask_shape, mask_dtype):
    s = rle_str.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    mask = np.zeros(np.prod(mask_shape), dtype=mask_dtype)
    for lo, hi in zip(starts, ends):
        mask[lo:hi] = 1
    return mask.reshape(mask_shape[::-1]).T


def probas_to_rles(x, cutoff=0.5):
    lab_img = label(x > cutoff)
    for i in range(1, lab_img.max() + 1):
        yield run_length_encode(lab_img == i)


def get_submit_data(predictions, ids):
    new_ids = []
    rles = []
    for n, id_ in enumerate(ids):
        rle = list(probas_to_rles(predictions[n]))
        rles.extend(rle)
        new_ids.extend([id_] * len(rle))
    return new_ids, rles
