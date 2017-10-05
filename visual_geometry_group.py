import numpy as np
import scipy.io
import tensorflow as tf

from logger import logger
from constants import VGG19_LAYERS


class VGG(object):
    """VGG provides an interface to extract parameter from pre-trained neural network
    and formulate Tensorflow layers"""
    def __init__(self, trained):
        network = scipy.io.loadmat(trained)
