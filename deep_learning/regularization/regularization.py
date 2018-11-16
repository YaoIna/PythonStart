import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import scipy.io

from deep_learning.initialization.init_utils import forward_propagation
from deep_learning.regularization.reg_utils import load_2d_data_set, compute_cost, initialize_parameters, \
    update_parameters, backward_propagation, predict, plot_decision_boundary, predict_dec, relu, sigmoid
from deep_learning.regularization.test_case import compute_cost_with_regularization_test_case, \
    backward_propagation_with_regularization_test_case, forward_propagation_with_dropout_test_case


def compute_cost_with_regularization(a3: np.ndarray, y: np.ndarray, parameters: dict, lam: float):
    W_sum = 0
    for k, v in parameters.items():
        if "W" in k:
            W_sum += np.sum(np.square(v))
    reg = lam * W_sum / (2 * y.shape[1])
    return compute_cost(a3, y) + reg


def backward_propagation_with_regularization(x: np.ndarray, y: np.ndarray, cache: list, lam: float):
    m = x.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache
    dZ3 = A3 - y
    dW3 = np.dot(dZ3, A2.T) / m + lam * W3 / m
    db3 = np.sum(dZ3, axis=1, keepdims=True) / m
    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = np.dot(dZ2, A1.T) / m + lam * W2 / m
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = np.dot(dZ1, x.T) / m + lam * W1 / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m
    return {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
            "dZ1": dZ1, "dW1": dW1, "db1": db1}


def model(x, y, learning_rate=0.3, num_iterations=30000, print_cost=True, lam=0.0, keep_prob=1.0):
    """
    Implements a three-layer neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SIGMOID.

    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (output size, number of examples)
    learning_rate -- learning rate of the optimization
    num_iterations -- number of iterations of the optimization loop
    print_cost -- If True, print the cost every 10000 iterations
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.

    Returns:
    parameters -- parameters learned by the model. They can then be used to predict.
    """

    grads = {}
    costs = []  # to keep track of the cost
    layers_dims = [x.shape[0], 20, 3, 1]

    # Initialize parameters dictionary.
    parameters = initialize_parameters(layers_dims)

    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.
        if keep_prob >= 1:
            a3, cache = forward_propagation(x, parameters)
        else:
            a3, cache = forward_propagation_with_dropout(x, parameters, keep_prob)
        # Cost function
        if lam == 0:
            cost = compute_cost(a3, y)
        else:
            cost = compute_cost_with_regularization(a3, y, parameters, lam)

        # Backward propagation.
        assert (lam == 0 or keep_prob == 1)  # it is possible to use both L2 regularization and dropout,
        # but this assignment will only explore one at a time
        if lam == 0 and keep_prob == 1:
            grads = backward_propagation(x, y, cache)
        elif lam != 0:
            grads = backward_propagation_with_regularization(x, y, cache, lam)
        elif keep_prob < 1:
            grads = backward_propagation_with_dropout(x, y, cache, keep_prob)

        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)

        # Print the loss every 10000 iterations
        if print_cost and i % 10000 == 0:
            print("Cost after iteration {}: {}".format(i, cost))
        if print_cost and i % 1000 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters


def forward_propagation_with_dropout(x, parameters, keep_prob=0.5):
    np.random.seed(1)

    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']

    Z1 = np.dot(W1, x) + b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0], A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1 / keep_prob
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0], A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2 / keep_prob
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)
    return A3, cache


def backward_propagation_with_dropout(x, y, cache, keep_prob):
    m = x.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
    dA2 = np.dot(W3.T, dZ3)
    dA2 = dA2 * D2  # Step 1: Apply mask D2 to shut down the same neurons as during the forward propagation
    dA2 = dA2 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dA1 = dA1 * D1  # Step 1: Apply mask D1 to shut down the same neurons as during the forward propagation
    dA1 = dA1 / keep_prob  # Step 2: Scale the value of neurons that haven't been shut down
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, x.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    return {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
            "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
            "dZ1": dZ1, "dW1": dW1, "db1": db1}


def plot_regularization_l2():
    train_x, train_y, test_x, test_y = load_2d_data_set()
    parameters = model(train_x, train_y, lam=0.7)
    print("On the train set:")
    predict(train_x, train_y, parameters)
    print("On the test set:")
    predict(train_x, train_y, parameters)

    plt.title("Model with L2-regularization")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)


def plot_drop_out():
    train_x, train_y, test_x, test_y = load_2d_data_set()

    parameters = model(train_x, train_y, keep_prob=0.86, learning_rate=0.3)
    print("On the train set:")
    predict(train_x, train_y, parameters)
    print("On the test set:")
    predict(test_x, test_y, parameters)

    plt.title("Model with dropout")
    axes = plt.gca()
    axes.set_xlim([-0.75, 0.40])
    axes.set_ylim([-0.75, 0.65])
    plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)


plot_drop_out()
