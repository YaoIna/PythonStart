import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage

from deep_learning.deep_neural_image.L_layer_nerual_network import l_layer_model
from deep_learning.deep_neural_image.dnn_app_utils import *
from deep_learning.deep_neural_image.two_layer_neural_network import two_layer_model

plt.rcParams['figure.figsize'] = (5.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)

train_x_orig, train_y, test_x_orig, test_y, classes = load_data()
train_x = train_x_orig.reshape(train_x_orig.shape[0], -1).T / 255
test_x = test_x_orig.reshape(test_x_orig.shape[0], -1).T / 255

# parameters = two_layer_model(train_x, train_y, (train_x.shape[0], 7, 1), num_iterations=2500, print_cost=True)
# predict_train = predict(train_x, train_y, parameters)
# predictions_test = predict(test_x, test_y, parameters)

layers_dims = [12288, 20, 7, 5, 1]

parameters = l_layer_model(train_x, train_y, layers_dims, num_iterations=2500, print_cost=True)

predictions_train = predict(train_x, train_y, parameters)

predictions_test = predict(test_x, test_y, parameters)
