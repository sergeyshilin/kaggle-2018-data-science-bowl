from __future__ import print_function

import os
os.environ['KERAS_BACKEND'] = 'tensorflow'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import numpy as np
import pandas as pd
import tensorflow as tf

import params
from augmentation import deterministic_augmentation
from tqdm import tqdm
import generators

from utils import get_data_train, get_data_test, get_best_history
from utils import get_predictions_upsampled, get_submit_data, probas_to_rles

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import log_loss, accuracy_score
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

### LOAD PARAMETERS
train_val_generator = generators.shilin_train_val_generator
#
epochs = params.max_epochs
batch_size = params.batch_size
best_weights_path = params.best_weights_path
best_weights_checkpoint = params.best_weights_checkpoint
best_model_path = params.best_model_path
init_weights = params.init_weights_path
num_folds = params.num_folds
tta_steps = params.tta_steps
model_input_size = params.model_input_size
transform_data = params.data_adapt
pseudolabeling = params.pseudolabeling
random_seed = params.seed
### LOAD PARAMETERS


X_train, y_train, train_ids, train_sizes = get_data_train('../data/stage1_train/', model_input_size)
X_test, test_ids, test_sizes = get_data_test('../data/stage1_test/', model_input_size)

def get_callbacks():
    return [
        EarlyStopping(monitor='val_loss', patience=20, verbose=1, min_delta=1e-4, mode='min'),
        ReduceLROnPlateau(monitor='val_loss', patience=10, factor=0.1, 
            verbose=1, epsilon=1e-4, mode='min'),
        ModelCheckpoint(monitor='val_loss', filepath=best_weights_checkpoint, 
            save_best_only=True, save_weights_only=True, mode='min')
    ]

model_info = params.model_factory(input_shape=X_train.shape[1:])
model_info.summary()
model_info.save_weights(filepath=init_weights)

with open(best_model_path, "w") as json_file:
    json_file.write(model_info.to_json())


def train_and_evaluate_model(model, xtr, ytr, xcv, ycv):
    train_generator, val_generator = train_val_generator(xtr, xcv, ytr, ycv)

    hist = model.fit_generator(
        train_generator,
        steps_per_epoch=np.ceil(float(len(xtr)) / float(batch_size)),
        epochs=epochs,
        verbose=1,
        validation_data=val_generator,
        validation_steps=np.ceil(float(len(xcv)) / float(batch_size)),
        callbacks=get_callbacks()
    )

    history = get_best_history(hist.history, monitor='val_loss', mode='min')
    best_epoch, loss, acc, val_loss, val_acc = history
    print ()
    print ("Best epoch: {}".format(best_epoch))
    print ("loss: {:0.6f} - acc: {:0.4f} - val_loss: {:0.6f} - val_acc: {:0.4f}".format(loss, acc, val_loss, val_acc))
    print ()
    return history


def predict_with_tta(model, X_data, verbose=0):
    predictions = np.zeros((tta_steps, X_data.shape[0], X_data.shape[1], X_data.shape[2], 1))
    
    for image_id in tqdm(range(X_data.shape[0])):
        test_probas = model.predict(X_data[np.newaxis, image_id, ...], batch_size=batch_size, verbose=verbose)
        predictions[0, image_id, ...] = test_probas

        for i in range(1, tta_steps):
            # perform augmentation and remember parameters
            image_aug, aug_params = deterministic_augmentation(X_data[image_id, ...], image = True) 
            # predict on augmented image
            test_probas = model.predict(image_aug[np.newaxis, ...], batch_size=batch_size, verbose=verbose)
            # perform inverse transform
            test_probas = deterministic_augmentation(test_probas[0], image = False, aug_params = aug_params)[0]
            predictions[i, image_id, ...] = test_probas

    predictions = predictions.mean(axis=0)
    return predictions


## ========================= RUN KERAS K-FOLD TRAINING ========================= ##
predictions = np.zeros((num_folds, len(X_test), model_input_size, model_input_size, 1))
tr_losses, tr_accs, val_losses, val_accs = [], [], [], []

skf = KFold(n_splits=num_folds, random_state=random_seed, shuffle=True)
for j, (train_index, cv_index) in enumerate(skf.split(X_train, y_train)):
    print ('\n===================FOLD=', j + 1)
    xtr, ytr = X_train[train_index], y_train[train_index]
    xcv, ycv = X_train[cv_index], y_train[cv_index]

    best_val_loss = np.inf
    best_val_accuracy = -1 * np.inf
    best_tr_loss = np.inf
    best_tr_accuracy = -1 * np.inf

    for lr in params.learning_rates:
        model_lr = None
        model_lr = params.model_factory(input_shape=X_train.shape[1:])
        model_lr.load_weights(filepath=init_weights)
        K.set_value(model_lr.optimizer.lr, lr)

        best_epoch, loss, acc, val_loss, val_acc = train_and_evaluate_model(model_lr, xtr, ytr, xcv, ycv)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_acc
            best_tr_loss = loss
            best_tr_accuracy = acc

            model_lr.load_weights(filepath=best_weights_checkpoint)
            model_lr.save_weights(filepath=best_weights_path)

    tr_losses.append(best_tr_loss)
    tr_accs.append(best_tr_accuracy)
    val_losses.append(best_val_loss)
    val_accs.append(best_val_accuracy)

    # Load the best model over all learning rates
    best_model = params.model_factory(input_shape=X_train.shape[1:])
    best_model.load_weights(filepath=best_weights_path)

    print ('\nPredicting test data with augmentation ...')
    fold_predictions = predict_with_tta(best_model, X_test, verbose=0)
    predictions[j] = fold_predictions

print ()
print ("Overall score: ")
print ("train_loss: {:0.6f} - train_acc: {:0.4f} - val_loss: {:0.6f} - val_acc: {:0.4f}".format(
    np.mean(tr_losses), np.mean(tr_accs), np.mean(val_losses), np.mean(val_accs)))
print ()

## ========================= MAKE CV AND LB SUBMITS ========================= ##
with open('submit_id', 'r') as submit_id:
    last_submit_id = int(submit_id.read())

last_submit_id += 1

new_test_ids, test_rles = get_submit_data(
    get_predictions_upsampled(predictions.mean(axis=0), test_sizes),
    test_ids
)

submission = pd.DataFrame()
submission['ImageId'] = new_test_ids
submission['EncodedPixels'] = pd.Series(test_rles)
submission.to_csv('../submits/submission_{0:0>3}.csv'.format(last_submit_id), index=False)

with open('submit_id', 'w') as submit_id:
    submit_id.write(str(last_submit_id))
