import os
import numpy as np
import skimage
from skimage.transform import resize
from skimage.morphology import label

from config import Config
import utils

##
## Global vars
##

init_with = "last"  # `imagenet`, `coco`, or `last`
layers_to_train = "all" # `heads` or `all`
epochs = 25
validation_split = 0.15


class BowlConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "nuclei" 
    GPU_COUNT = 1 
    IMAGES_PER_GPU = 1 
    NUM_CLASSES = 1 + 1 
    IMAGE_MIN_DIM = 256 
    IMAGE_MAX_DIM = 512 
    RPN_ANCHOR_SCALES = (4, 8, 16, 32, 64) 
    TRAIN_ROIS_PER_IMAGE = 500 
    STEPS_PER_EPOCH = 600 // (IMAGES_PER_GPU * GPU_COUNT) 
    VALIDATION_STEPS = 70 // (IMAGES_PER_GPU * GPU_COUNT) 
    MEAN_PIXEL = [0, 0, 0] 
    LEARNING_RATE = 0.001 
    USE_MINI_MASK = True 
    MAX_GT_INSTANCES = 500


class BowlDataset(utils.Dataset):
    """Generates the shapes synthetic dataset. The dataset consists of simple
    shapes (triangles, squares, circles) placed randomly on a blank surface.
    The images are generated on the fly. No file access required.
    """

    def load_shapes(self, data_path, ids, height, width):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        self.id_names = ids
        self.data_path = data_path
        # Add classes
        self.add_class("nuclei", 1, "nucleus")

        # Add images
        # Generate random specifications of images (i.e. color and
        # list of shapes sizes and locations). This is more compact than
        # actual images. Images are generated on the fly in load_image().
        for id_ in ids:
            img_path = '{}/{}/images/{}.png'.format(self.data_path, id_, id_)
            self.add_image("nuclei", image_id=id_, path=img_path,
                           height=height, width=width)
    
    def load_image(self, image_id, train = True):
        """Load the specified image and return a [H,W,3] Numpy array.
        """
        # Load image
        info = self.image_info[image_id]
        # print(info)
        image = skimage.io.imread(info['path'])[:,:,:3]
        # we only need to resize in case we train
        # if we test, we will resize in get_image_gt, but will keep the previous shape
        if train:
            image = resize(image, (info['height'], info['width']), mode='constant', preserve_range=True)
        # If grayscale. Convert to RGB for consistency.
        if image.ndim != 3:
            image = skimage.color.gray2rgb(image)
        return image

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        
        masks_path = '{}/{}/masks/'.format(self.data_path, self.id_names[image_id])
        masks_ids = np.asarray(os.listdir(masks_path))
        number_of_masks = len(masks_ids)
        
        info = self.image_info[image_id]
        mask = np.zeros((info['height'], info['width'], number_of_masks), dtype=np.uint8)
        for i, mask_id in enumerate(masks_ids):
            single_mask_path = masks_path + '/' + mask_id
            mask_ = skimage.io.imread(single_mask_path)
            mask_ = np.expand_dims(resize(
                    mask_, (info['height'], info['width']),
                    mode='constant', preserve_range=True
                ),
                axis=-1
            )
            mask[:, :, i:i+1] = mask_

        # Map class names to class IDs.
        class_ids = np.asarray([self.class_names.index('nucleus') for s in range(number_of_masks)])
        return mask, class_ids.astype(np.int32)

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "nuclei":
            return info["nuclei"]
        else:
            super(self.__class__).image_reference(self, image_id)

##
## Utils
##

def probas_to_rles(lab_img, cutoff=0.5):
    for i in range(1, lab_img.max() + 1):
        return run_length_encode(lab_img == i)


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
