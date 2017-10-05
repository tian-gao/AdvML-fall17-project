# path
PATH_INPUT_STYLE = 'input/style/'
PATH_INPUT_CONTENT = 'input/content/'
PATH_OUTPUT = 'output/'

# pre-trained network data
TRAINED_NETWORK_DATA = 'imagenet-vgg-verydeep-19.mat'

# VGG layder definition
VGG19_LAYERS = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4'
)

# input image layer mapping
CONTENT_LAYERS = ('relu4_2', 'relu5_2')
STYLE_LAYERS = ('relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1')

# default parameters
CONTENT_WEIGHT = 5
CONTENT_WEIGHT_BLEND = 1
STYLE_WEIGHT = 500
TV_WEIGHT = 100
STYLE_LAYER_WEIGHT_EXP = 1
LEARNING_RATE = 10
BETA1 = 0.9
BETA2 = 0.999
EPSILON = 1e-08
STYLE_SCALE = 1.0
ITERATIONS = 1000
POOLING = 'max'
