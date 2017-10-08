import os
import sys
import time
import scipy.misc
from argparse import ArgumentParser
from utils import read_image, save_image

from logger import logger
from settings import PATH_INPUT_STYLE, PATH_INPUT_CONTENT, PATH_OUTPUT, TRAINED_NETWORK_DATA
from constants import (
    CONTENT_WEIGHT, STYLE_WEIGHT, TV_WEIGHT, POOLING,
    LEARNING_RATE, BETA1, BETA2, EPSILON, MAX_ITERATION
)
from visual_geometry_group import VGG
from neural_network import NeuralNetwork


def style_transfer(
        content_name, style_name, output_name, content_weight, style_weight, tv_weight,
        pooling, learning_rate, beta1, beta2, epsilon, max_iteration, check_point):
    time_start = time.time()

    # read images
    content = read_image(PATH_INPUT_CONTENT + content_name)
    style = read_image(PATH_INPUT_STYLE + style_name)
    style = scipy.misc.imresize(style, content.shape[1] / style.shape[1])

    # initialize objects
    vgg = VGG(TRAINED_NETWORK_DATA, pooling)
    nn = NeuralNetwork(content, style, vgg, content_weight, style_weight, tv_weight)

    # train model
    for k, output_image in nn.train_model(learning_rate, beta1, beta2, epsilon, max_iteration, check_point):
        name_list = output_name.split('.')
        image_name = PATH_OUTPUT + '.'.join(name_list[:-1]) + '_{}.{}'.format(str(k) if k >= 0 else 'final', name_list[-1])
        save_image(output_image, image_name)

    time_end = time.time()
    logger.info('Time elapsed: {} seconds'.format(round(time_end - time_start)))


def build_parser():
    parser = ArgumentParser()
    parser.add_argument('--content', dest='content', required=True,
                        help='Content image, e.g. "input.jpg"')
    parser.add_argument('--style', dest='style', required=True,
                        help='Style image, e.g. "style.jpg"')
    parser.add_argument('--output', dest='output', required=True,
                        help='Output image, e.g. "output.jpg"')
    return parser


if __name__ == '__main__':
    parser = build_parser()
    args = parser.parse_args()

    # check if network data file exists
    if not os.path.isfile(TRAINED_NETWORK_DATA):
        logger.error('Cannot find pre-trained network data file!')
        sys.exit()

    style_transfer(
        content_name=args.content,
        style_name=args.style,
        output_name=args.output,

        content_weight=CONTENT_WEIGHT,
        style_weight=STYLE_WEIGHT,
        tv_weight=TV_WEIGHT,
        pooling=POOLING,

        learning_rate=LEARNING_RATE,
        beta1=BETA1,
        beta2=BETA2,
        epsilon=EPSILON,
        max_iteration=MAX_ITERATION,
        check_point=MAX_ITERATION / 10
    )
