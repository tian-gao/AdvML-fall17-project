import numpy as np
import scipy.misc
from PIL import Image

from logger import logger


def process_image(input_image, mean_pixel):
    return input_image - mean_pixel


def unprocess_image(input_image, mean_pixel):
    return input_image + mean_pixel


def read_image(image_name):
    logger.info('Reading image: {}'.format(image_name.split('/')[-1]))
    img = scipy.misc.imread(image_name).astype(np.float)
    img = img[:, :, :3]
    return img


def save_image(output_image, image_name):
    logger.info('Saving image: {}'.format(image_name.split('/')[-1]))
    img = np.clip(output_image, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(image_name, quality=95)
