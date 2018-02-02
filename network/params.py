from models import uNet_256

from utils.data_preprocessing import identical

seed = 13
max_epochs = 100
batch_size = 16
model_factory = uNet_256

data_adapt = identical
pseudolabeling = False

model_input_size = (256, 256, 3)
best_weights_path = 'weights/best_weights.hdf5'
best_weights_checkpoint = 'weights/best_weights_checkpoint.hdf5'
init_weights_path = 'weights/init_weights.hdf5'
best_model_path = 'models/best_model.json'
tta_steps = 10
num_folds = 5
learning_rates = [0.001 * 3., 0.001, 0.001 / 3.]

## Augmentation parameters
aug_horizontal_flip = False
aug_vertical_flip = False
aug_rotation = 0.0
aug_width_shift = 0.0
aug_height_shift = 0.0
aug_channel_shift = 0.2
aug_shear = 0.05
aug_zoom = 0.0
