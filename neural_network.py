import numpy as np
import tensorflow as tf
from functools import reduce
from operator import mul

from logger import logger
from constants import CONTENT_LAYERS, STYLE_LAYERS
from utils import process_image, unprocess_image


class NeuralNetwork(object):
    """NeuralNetwork provides an interface to formulate the Tensorflow neural network model
    and perform style transfer algorithm"""
    def __init__(self, content, style, vgg, content_weight, style_weight, tv_weight):
        logger.info('Initializing neural network......')
        self.content = content
        self.style = style
        self.vgg = vgg

        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        self.content_shape, self.style_shape, self.content_layer_weights, self.style_layer_weights = self.get_parameters()
        self.content_features, self.style_features = self.get_features()

    def get_parameters(self):
        logger.info('Fetching images parameters......')
        content_shape = (1, ) + self.content.shape
        style_shape = (1, ) + self.style.shape

        # get content layer weights
        content_layer_weights = {}
        content_layer_weights['relu4_2'] = 1.0
        content_layer_weights['relu5_2'] = 0.0

        # get style layer weights
        style_layer_weights = {}
        for style_layer in STYLE_LAYERS:
            style_layer_weights[style_layer] = 1.0 / len(STYLE_LAYERS)

        return content_shape, style_shape, content_layer_weights, style_layer_weights

    def get_features(self):
        content_features = self._get_content_feature()
        style_features = self._get_style_feature()
        return content_features, style_features

    def _get_content_feature(self):
        logger.info('Fetching content features......')
        content_features = {}
        graph = tf.Graph()
        with graph.as_default(), graph.device('/cpu:0'), tf.Session() as session:
            content_image = tf.placeholder('float', shape=self.content_shape)
            content_net = self.vgg.load_net(content_image)
            content_pre = np.array([
                process_image(self.content, self.vgg.mean_pixel)])
            for content_layer in CONTENT_LAYERS:
                content_features[content_layer] = content_net[content_layer].eval(feed_dict={content_image: content_pre})

        return content_features

    def _get_style_feature(self):
        logger.info('Fetching style features......')
        style_features = {}
        graph = tf.Graph()
        with graph.as_default(), graph.device('/cpu:0'), tf.Session() as session:
            style_image = tf.placeholder('float', shape=self.style_shape)
            style_net = self.vgg.load_net(style_image)
            style_pre = np.array([
                process_image(self.style, self.vgg.mean_pixel)])
            for style_layer in STYLE_LAYERS:
                feature = style_net[style_layer].eval(feed_dict={style_image: style_pre})
                feature = np.reshape(feature, (-1, feature.shape[3]))
                gram = feature.T.dot(feature) / feature.size
                style_features[style_layer] = gram

        return style_features

    def train_model(self, learning_rate, beta1, beta2, epsilon, max_iteration):
        with tf.Graph().as_default():
            # initialize with random guess
            logger.info('Initializing tensorflow graph with random guess......')
            noise = np.random.normal(size=self.content_shape, scale=np.std(self.content) * 0.1)
            initial_guess = tf.random_normal(self.content_shape) * 0.256
            input_image = tf.Variable(initial_guess)
            parsed_net = self.vgg.load_net(input_image)

            # calculate loss
            content_loss = self._calculate_content_loss(parsed_net)
            style_loss = self._calculate_style_loss(parsed_net)
            tv_loss = self._calculate_tv_loss(input_image)
            loss = content_loss + style_loss + tv_loss

            # initialize optimization
            train_step = tf.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)
            best_loss = float('inf')

            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                logger.info('Initializing optimization......')
                logger.info('Current total loss: {}'.format(loss.eval()))

                for k in range(max_iteration):
                    logger.info('Iteration {} total loss {}'.format(str(k+1), loss.eval()))
                    train_step.run()

                # form output image
                output_temp = input_image.eval()
                output_image = unprocess_image(output_temp.reshape(self.content_shape[1:]), self.vgg.mean_pixel)

                # yield output_image
                return output_image

    def _calculate_content_loss(self, parsed_net):
        logger.info('Calculating content loss......')
        losses = []
        for content_layer in CONTENT_LAYERS:
            losses += [
                self.content_layer_weights[content_layer] * self.content_weight * (
                    2 * tf.nn.l2_loss(
                        parsed_net[content_layer] - self.content_features[content_layer]
                    ) / self.content_features[content_layer].size)]
        return reduce(tf.add, losses)

    def _calculate_style_loss(self, parsed_net):
        logger.info('Calculating style loss......')
        losses = []
        for style_layer in STYLE_LAYERS:
            layer = parsed_net[style_layer]
            _, height, width, number = map(lambda x: x.value, layer.get_shape())
            size = height * width * number
            feats = tf.reshape(layer, (-1, number))
            gram = tf.matmul(tf.transpose(feats), feats) / size
            style_gram = self.style_features[style_layer]
            losses += [
                self.style_layer_weights[style_layer] * 2 * tf.nn.l2_loss(gram - style_gram) / style_gram.size]
        return self.style_weight * reduce(tf.add, losses)

    def _calculate_tv_loss(self, image):
        # total variation denoising
        logger.info('Calculating total variation loss......')
        tv_y_size = self._get_tensor_size(image[:, 1:, :, :])
        tv_x_size = self._get_tensor_size(image[:, :, 1:, :])
        tv_loss = self.tv_weight * 2 * ((
            tf.nn.l2_loss(image[:, 1:, :, :] - image[:, :self.content_shape[1]-1, :, :]) / tv_y_size) + (
            tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, :self.content_shape[2]-1, :]) / tv_x_size))
        return tv_loss

    def _get_tensor_size(self, tensor):
        return reduce(mul, (d.value for d in tensor.get_shape()), 1)
