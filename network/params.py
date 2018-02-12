from models import uNet_256
from utils import DataPreprocessing

seed = 13
max_epochs = 100
batch_size = 16
model_factory = uNet_256

data_adapt = DataPreprocessing.identical
pseudolabeling = False

model_input_size = 256
best_weights_path = 'weights/best_weights.hdf5'
best_weights_checkpoint = 'weights/best_weights_checkpoint.hdf5'
init_weights_path = 'weights/init_weights.hdf5'
best_model_path = 'models/best_model.json'
tta_steps = 1 # original image only, no additional augmentation 
num_folds = 5
learning_rates = [0.001 * 3., 0.001, 0.001 / 3.]

## new augmentation parameters
aug_flip_proba = 0.5
aug_max_shift = 0.5
aug_max_angle = 15
aug_num_transforms = 4
aug_possible_transforms = ['flip_0', 'flip_1', 'shift_0', 'shift_1', 'rotate']

# old augmentation parameters
aug_horizontal_flip = True
aug_vertical_flip = True
aug_rotation = 30
aug_width_shift = 0.1
aug_height_shift = 0.1
aug_channel_shift = 0.0
aug_shear = 0.0
aug_zoom = 0.1
aug_fill_mode = 'reflect'
