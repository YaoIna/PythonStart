import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets

from deep_learning.initialization.init_utils import load_data_set, forward_propagation, compute_loss, \
    backward_propagation, update_parameters, predict, plot_decision_boundary, predict_dec

plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


def model(x, y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he"):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (containing 0 for red dots; 1 for blue dots), of shape (1, number of examples)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations to run gradient descent
    print_cost -- if True, print the cost every 1000 iterations
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")

    Returns:
    parameters -- parameters learnt by the model
    """

    costs = []  # to keep track of the loss
    layers_dims = [x.shape[0], 10, 5, 1]

    # Initialize parameters dictionary.
    if initialization == "zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization == "random":
        parameters = initialize_parameters_random(layers_dims)
    else:
        parameters = initialize_parameters_he(layers_dims)
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        a3, cache = forward_propagation(x, parameters)

        # Loss
        cost = compute_loss(a3, y)

        # Backward propagation.
        grads = backward_propagation(x, y, cache)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
            costs.append(cost)

    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def initialize_parameters_zeros(layers_dims: list) -> dict:
    parameters = {}
    for i in range(1, len(layers_dims)):
        parameters['W%d' % i] = np.zeros(shape=(layers_dims[i], layers_dims[i - 1]))
        parameters['b%d' % i] = np.zeros(shape=(layers_dims[i], 1))
    return parameters


def initialize_parameters_random(layers_dims: list) -> dict:
    np.random.seed(3)
    parameters = {}
    for i in range(1, len(layers_dims)):
        parameters['W%d' % i] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * 10
        parameters['b%d' % i] = np.zeros(shape=[layers_dims[i], 1])
    return parameters


def initialize_parameters_he(layers_dims: list) -> dict:
    np.random.seed(3)
    parameters = {}
    for i in range(1, len(layers_dims)):
        parameters['W%d' % i] = np.random.randn(layers_dims[i], layers_dims[i - 1]) * np.sqrt(2 / layers_dims[i - 1])
        parameters['b%d' % i] = np.zeros(shape=[layers_dims[i], 1])
    return parameters


def plot_initialization_zeros():
    train_x, train_y, test_x, test_y = load_data_set()

    parameters = model(train_x, train_y, initialization='zeros')
    print("On the train set:")
    predict(train_x, train_y, parameters)
    print("On the test set:")
    predict(test_x, test_y, parameters)

    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)


def plot_initialization_random():
    train_x, train_y, test_x, test_y = load_data_set()
    parameters = model(train_x, train_y, initialization="random")
    print("On the train set:")
    predict(train_x, train_y, parameters)
    print("On the test set:")
    predict(train_x, train_y, parameters)

    plt.title("Model with large random initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)


def plot_initialization_he():
    train_x, train_y, test_x, test_y = load_data_set()
    parameters = model(train_x, train_y, initialization="he")
    print("On the train set:")
    predict(train_x, train_y, parameters)
    print("On the test set:")
    predict(train_x, train_y, parameters)

    plt.title("Model with large he initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)


plot_initialization_he()
