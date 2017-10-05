import numpy as np
import tensorflow as tf
from PIL import Image

from logger import logger
from settings import PATH_INPUT_STYLE, PATH_INPUT_CONTENT, PATH_OUTPUT
from constants import CONTENT_LAYERS, STYLE_LAYERS


class NeuralNetwork(object):
    """NeuralNetwork provides an interface to formulate the Tensorflow neural network model
    and perform style transfer algorithm"""
    def __init__(self, content, style):
        network = scipy.io.loadmat(trained)
