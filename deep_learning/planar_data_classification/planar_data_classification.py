import numpy as np
import matplotlib.pyplot as plt
from deep_learning.planar_data_classification.planar_utils import plot_decision_boundary, sigmoid, load_planar_data_set

np.random.seed(1)
x, y = load_planar_data_set()


def layer_sizes(input_x: np.ndarray, output_y: np.ndarray):
    """
    Arguments:
    input_x -- input dataset of shape (input size, number of examples)
    output_y -- labels of shape (output size, number of examples)

    Returns:
    n_x -- the size of the input layer
    n_y -- the size of the output layer
    """
    n_x = input_x.shape[0]  # size of input layer
    n_y = output_y.shape[0]  # size of output layer
    return n_x, n_y


def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)
    w1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(shape=(n_h, 1))
    w2 = np.random.rand(n_y, n_h) * 0.01
    b2 = np.zeros(shape=(n_y, 1))
    assert (w1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (w2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def forward_propagation(x_data: np.ndarray, parameters: dict):
    w1 = parameters['w1']
    b1 = parameters['b1']
    w2 = parameters['w2']
    b2 = parameters['b2']

    z1 = np.dot(w1, x_data) + b1
    a1 = np.tanh(z1)
    z2 = np.dot(w2, a1) + b2
    a2 = sigmoid(z2)

    assert (a2.shape == (1, x_data.shape[1]))
    return {'z1': z1, 'a1': a1, 'z2': z2, 'a2': a2}


def compute_cost(y_hat: np.ndarray, y_data: np.ndarray):
    assert (y_hat.shape == y_data.shape)
    m = y_data.shape[1]
    if m == 0:
        print(y_data)

    cost = np.sum(-(np.multiply(y_data, np.log(y_hat)) + np.multiply(1 - y_data, np.log(1 - y_hat))), axis=1) / m
    cost = cost[0]
    assert (isinstance(cost, float))
    return cost


def backward_propagation(parameter: dict, cache: dict, x_data: np.ndarray, y_data: np.ndarray):
    m = x_data.shape[1]

    w2 = parameter['w2']
    a1 = cache['a1']
    a2 = cache['a2']

    dz2 = a2 - y_data
    dw2 = np.dot(dz2, a1.T) / m
    db2 = np.sum(dz2, axis=1, keepdims=True) / m
    dz1 = np.multiply(np.dot(w2.T, dz2), 1 - np.power(a1, 2))
    dw1 = np.dot(dz1, x_data.T) / m
    db1 = np.sum(dz1, axis=1, keepdims=True) / m
    return {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}


def update_parameters(parameters, grads, learning_rate=1.2):
    dw1 = grads['dw1']
    db1 = grads['db1']
    dw2 = grads['dw2']
    db2 = grads['db2']

    w1 = parameters['w1'] - dw1 * learning_rate
    b1 = parameters['b1'] - db1 * learning_rate
    w2 = parameters['w2'] - dw2 * learning_rate
    b2 = parameters['b2'] - db2 * learning_rate

    return {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}


def nn_model(x_data, y_data, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x, n_y = layer_sizes(x_data, y_data)
    param = initialize_parameters(n_x, n_h, n_y)
    for i in range(0, num_iterations):
        forward_result = forward_propagation(x_data, param)
        cost_value = compute_cost(forward_result['a2'], y_data)
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost_value))
        grads = backward_propagation(param, forward_result, x_data, y_data)
        param = update_parameters(param, grads)
    return param


def predict(parameters, x_data):
    result = forward_propagation(x_data, parameters)
    predictions = np.round(result['a2'])
    return predictions


data, label = load_planar_data_set()
plt.figure(figsize=(16, 32))
hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
for i, n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5, 2, i + 1)
    plt.title('Hidden Layer of size %d' % n_h)
    model_parameters = nn_model(data, label, n_h, num_iterations=5000)
    plot_decision_boundary(lambda x: predict(model_parameters, x.T), data, label)
    model_predictions = predict(model_parameters, data)
    accuracy = float(
        (np.dot(label, model_predictions.T) + np.dot(1 - label, 1 - model_predictions.T)) / float(label.size) * 100)
    print("Accuracy for {} hidden units: {} %".format(n_h, accuracy))
