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
learning_rates = [0.0001 * 3., 0.0001, 0.0001 / 3.]

## Augmentation parameters
aug_horizontal_flip = True
aug_vertical_flip = True
aug_rotation = 0.30
aug_width_shift = 0.2
aug_height_shift = 0.2
aug_channel_shift = 0.2
aug_shear = 0.3
aug_zoom = 0.2
aug_fill_mode = 'reflect'
