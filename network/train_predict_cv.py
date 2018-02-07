from __future__ import print_function

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf

import params
from utils import get_data_train, get_data_test, get_best_history
from utils import get_predictions_upsampled, get_submit_data, probas_to_rles
from losses import bce_dice_loss_list, mean_iou

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K


### LOAD PARAMETERS
epochs = params.max_epochs
batch_size = params.batch_size
best_weights_path = params.best_weights_path
best_weights_checkpoint = params.best_weights_checkpoint
best_model_path = params.best_model_path
init_weights = params.init_weights_path
random_seed = params.seed
num_folds = params.num_folds
tta_steps = params.tta_steps
model_input_size = params.model_input_size
transform_data = params.data_adapt
pseudolabeling = params.pseudolabeling


## Augmentation parameters
aug_horizontal_flip = params.aug_horizontal_flip
aug_vertical_flip = params.aug_vertical_flip
aug_rotation = params.aug_rotation
aug_width_shift = params.aug_width_shift
aug_height_shift = params.aug_height_shift
aug_channel_shift = params.aug_channel_shift
aug_shear = params.aug_shear
aug_zoom = params.aug_zoom
### LOAD PARAMETERS


X_train, y_train, train_ids, train_sizes = get_data_train('../data/train/', model_input_size)
X_test, test_ids, test_sizes = get_data_test('../data/test/', model_input_size)


def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=40, verbose=1, min_delta=1e-4, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', patience=20, factor=0.1, 
            verbose=1, epsilon=1e-4, mode='min'),
        ModelCheckpoint(monitor='val_loss', filepath=best_weights_checkpoint, 
            save_best_only=True, save_weights_only=True, mode='min')
    ]


datagen_args = dict(
    horizontal_flip=aug_horizontal_flip,
    vertical_flip=aug_vertical_flip,
    rotation_range=aug_rotation,
    width_shift_range=aug_width_shift,
    height_shift_range=aug_height_shift,
    channel_shift_range=aug_channel_shift,
    shear_range=aug_shear,
    zoom_range=aug_zoom
)


def generator(xtr, xval, ytr, yval):
    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)
    image_datagen.fit(xtr, seed=random_seed)
    mask_datagen.fit(ytr, seed=random_seed)
    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=random_seed)
    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=random_seed)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args)
    mask_datagen_val = ImageDataGenerator(**val_gen_args)
    image_datagen_val.fit(xval, seed=random_seed)
    mask_datagen_val.fit(yval, seed=random_seed)
    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=random_seed)
    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=random_seed)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator


model_info = params.model_factory(input_shape=X_train.shape[1:])
model_info.summary()
model_info.save_weights(filepath=init_weights)

with open(best_model_path, "w") as json_file:
    json_file.write(model_info.to_json())


def train_and_evaluate_model(model, xtr, ytr, xcv, ycv):
    train_generator, val_generator = generator(xtr, xcv, ytr, ycv)

    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(xtr)) / float(batch_size)),
        epochs=epochs,
        verbose=1,
        validation_data=val_generator,
        validation_steps=np.ceil(float(len(xcv)) / float(batch_size)),
        callbacks=get_callbacks()
    )

    best_epoch, loss, acc, val_loss, val_acc = get_best_history(hist.history, monitor='val_loss', mode='min')
    print ()
    print ("Best epoch: {}".format(best_epoch))
    print ("loss: {:0.6f} - acc: {:0.4f} - val_loss: {:0.6f} - val_acc: {:0.4f}".format(loss, acc, val_loss, val_acc))
    print ()
    return val_loss


def predict_with_tta(model, X_data, verbose=0):
    predictions = np.zeros((tta_steps, X_data.shape[0], X_data.shape[1], X_data.shape[2], 1))
    test_probas = model.predict(X_data, batch_size=batch_size, verbose=verbose)
    predictions[0] = test_probas

    for i in range(1, tta_steps):
        test_probas = model.predict_generator(
            get_data_generator_test(datagen, X_data, M_data, batch_size=batch_size),
            steps=np.ceil(float(len(X_data)) / float(batch_size)),
            verbose=verbose
        )
        predictions[i] = test_probas

    predictions = predictions.mean(axis=0)
    return predictions


## ========================= RUN KERAS K-FOLD TRAINING ========================= ##
predictions = np.zeros((num_folds, len(X_test), model_input_size, model_input_size, 1))
cv_labels = np.zeros((len(X_train), model_input_size, model_input_size, 1), dtype=np.uint8)
cv_preds = np.zeros((len(X_train), model_input_size, model_input_size, 1), dtype=np.float32)
# tr_labels, tr_preds = [], []

skf = KFold(n_splits=num_folds, random_state=random_seed, shuffle=True)
for j, (train_index, cv_index) in enumerate(skf.split(X_train, y_train)):
    print ('\n===================FOLD=', j + 1)
    xtr, ytr = X_train[train_index], y_train[train_index]
    xcv, ycv = X_train[cv_index], y_train[cv_index]

    # tr_labels.extend(ytr)
    cv_labels[cv_index] = ycv

    best_val_loss = 100000.0

    for lr in params.learning_rates:
        model_lr = None
        model_lr = params.model_factory(input_shape=X_train.shape[1:])
        model_lr.load_weights(filepath=init_weights)
        K.set_value(model_lr.optimizer.lr, lr)

        val_loss = train_and_evaluate_model(model_lr, xtr, ytr, xcv, ycv)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            model_lr.load_weights(filepath=best_weights_checkpoint)
            model_lr.save_weights(filepath=best_weights_path)

    # Load the best model over all learning rates
    best_model = params.model_factory(input_shape=X_train.shape[1:])
    best_model.load_weights(filepath=best_weights_path)

    # Measure train and validation quality
    print ('\nValidating accuracy on training data ...')
    # tr_preds.extend(predict_with_tta(best_model, xtr))
    cv_preds[cv_index] = predict_with_tta(best_model, xcv)

    print ('\nPredicting test data with augmentation ...')
    fold_predictions = predict_with_tta(best_model, X_test, verbose=1)
    predictions[j] = fold_predictions

# tr_loss = bce_dice_loss_list(tr_labels, tr_preds)
# tr_acc = mean_iou_list(tr_labels, tr_preds)
sess = tf.Session()
with tf.sess.as_default():
    val_loss = bce_dice_loss_list(cv_labels, cv_preds)
    val_acc = mean_iou(cv_labels, cv_preds)

print ()
print ("Overall score: ")
# print ("train_loss: {:0.6f} - train_acc: {:0.4f} - val_loss: {:0.6f} - val_acc: {:0.4f}".format(
#     tr_loss, tr_acc, val_loss, val_acc))
print ("val_loss: {:0.6f} - val_acc: {:0.4f}".format(val_loss, val_acc))
print ()

## ========================= MAKE CV AND LB SUBMITS ========================= ##
with open('submit_id', 'r') as submit_id:
    last_submit_id = int(submit_id.read())

last_submit_id += 1

# submission_cv = pd.DataFrame()
# submission_cv['preds'] = get_preds_upsampled(cv_preds, train_sizes)
# submission_cv['is_iceberg'] = cv_labels
# submission_cv.to_csv('../submits_cv/submission_cv_{0:0>3}.csv'.format(last_submit_id), index=False)

new_test_ids, test_rles = get_submit_data(
    get_predictions_upsampled(predictions.mean(axis=0), test_sizes),
    test_ids
)

submission = pd.DataFrame()
submission['ImageId'] = new_test_ids
submission['EncodedPixels'] = pd.Series(test_rles).apply(lambda x: ' '.join(str(y) for y in x))
submission.to_csv('../submits/submission_{0:0>3}.csv'.format(last_submit_id), index=False)

with open('submit_id', 'w') as submit_id:
    submit_id.write(str(last_submit_id))
