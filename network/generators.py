from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import params
from augmentation import random_augmentation, deterministic_augmentation

batch_size = params.batch_size
random_seed = params.seed

datagen_args = dict(
    horizontal_flip=params.aug_horizontal_flip,
    vertical_flip=params.aug_vertical_flip,
    rotation_range=params.aug_rotation,
    width_shift_range=params.aug_width_shift,
    height_shift_range=params.aug_height_shift,
    channel_shift_range=params.aug_channel_shift,
    shear_range=params.aug_shear,
    zoom_range=params.aug_zoom,
    fill_mode=params.aug_fill_mode
)


def shilin_train_val_generator(xtr, xval, ytr, yval):
    image_datagen = ImageDataGenerator(**datagen_args)
    mask_datagen = ImageDataGenerator(**datagen_args)

    image_datagen.fit(xtr, seed=random_seed)
    mask_datagen.fit(ytr, seed=random_seed)

    image_generator = image_datagen.flow(xtr, batch_size=batch_size, seed=random_seed)
    mask_generator = mask_datagen.flow(ytr, batch_size=batch_size, seed=random_seed)
    train_generator = zip(image_generator, mask_generator)

    val_gen_args = dict()
    image_datagen_val = ImageDataGenerator(**val_gen_args) # no augmentation
    mask_datagen_val = ImageDataGenerator(**val_gen_args) # no augmentation

    image_datagen_val.fit(xval, augment=True, seed=random_seed)
    mask_datagen_val.fit(yval, augment=True, seed=random_seed)

    image_generator_val = image_datagen_val.flow(xval, batch_size=batch_size, seed=random_seed)
    mask_generator_val = mask_datagen_val.flow(yval, batch_size=batch_size, seed=random_seed)
    val_generator = zip(image_generator_val, mask_generator_val)

    return train_generator, val_generator

def astra_train_val_generator(xtr, val, ytr, yval):
    def astra_train_generator(xtr, ytr):
        idxs = np.arange(xtr.shape[0])
        np.random.shuffle(idxs)
        
        x_batch = []
        y_batch = []
        idx = 0
        while True:
            img, mask = random_augmentation(xtr[idx], ytr[idx])
            idx += 1
            x_batch.append(img)
            y_batch.append(mask)
            if len(x_batch) == batch_size:
                yield np.array(x_batch, np.float32), np.array(y_batch, np.float32)
                x_batch = []
                y_batch = []
            
            if idx >= xtr.shape[0]:
                idx = 0
    return astra_train_generator(xtr, ytr), astra_train_generator(val, yval)
 
