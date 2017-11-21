import numpy as np
import scipy.io
import tensorflow as tf

from logger import logger
from constants import VGG19_LAYERS


class VGG(object):
    """VGG provides an interface to extract parameter from pre-trained neural network
    and formulate Tensorflow layers"""
    def __init__(self, trained, pooling):
        logger.info('Loading pre-trained network data......')
        self.network = scipy.io.loadmat(trained)
        self.layers, self.mean_pixel = self.init_net()
        self.pooling = pooling

    def init_net(self):
        mean_mat = self.network['normalization'][0][0][0]  # shape: (224, 224, 3)
        mean_pixel = np.mean(mean_mat, axis=(0, 1))  # length: 3
        layers = self.network['layers'].reshape(-1)  # length: 43
        return layers, mean_pixel

    def load_net(self, input_image):
        # construct layers using parameters
        logger.info('Parsing layers......')
        parsed_net = {}
        current_image = input_image

        for layer_name, input_layer in zip(VGG19_LAYERS, self.layers):
            layer_kind = layer_name[:4]

            if layer_kind == 'conv':
                current_image = self._get_conv_layer(current_image, input_layer)
            elif layer_kind == 'relu':
                current_image = self._get_relu_layer(current_image)
            elif layer_kind == 'pool':
                current_image = self._get_pool_layer(current_image)
            parsed_net[layer_name] = current_image

        assert len(parsed_net) == len(VGG19_LAYERS)
        return parsed_net

    def _get_conv_layer(self, input_image, input_layer):
        # get kernel and bias
        kernels, bias = input_layer[0][0][0][0]
        kernels = np.transpose(kernels, (1, 0, 2, 3))
        bias = bias.reshape(-1)

        # formulate conv layer
        conv = tf.nn.conv2d(input_image, tf.constant(kernels), strides=(1, 1, 1, 1), padding='SAME')
        layer = tf.nn.bias_add(conv, bias)
        return layer

    def _get_relu_layer(self, input_image):
        return tf.nn.relu(input_image)

    def _get_pool_layer(self, input_image):
        if self.pooling == 'avg':
            layer = tf.nn.avg_pool(input_image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        elif self.pooling == 'max':
            layer = tf.nn.max_pool(input_image, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')
        return layer
