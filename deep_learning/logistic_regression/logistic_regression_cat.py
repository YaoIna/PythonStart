from deep_learning import lr_utils
import numpy as np
import matplotlib.pyplot as plt

from deep_learning.python_basics_with_numpy import sigmoid

train_x_orig, train_y, test_x_orig, test_y, classes = lr_utils.load_data_set()
train_x_flatten = train_x_orig.reshape(train_x_orig.shape[0], -1).T
test_x_flatten = test_x_orig.reshape(test_x_orig.shape[0], -1).T

# standardize the data
train_x = train_x_flatten / 255
test_x_orig = test_x_flatten / 255


def initialize_with_zeros(dim: int):
    w = np.zeros(shape=(dim, 1))
    b = 0

    assert (w.shape == (dim, 1))
    assert (isinstance(b, float) or isinstance(b, int))
    return w, b


def propagate(w: np.ndarray, b, x: np.ndarray, y: np.ndarray):
    m = x.shape[1]
    y_hat = sigmoid(np.dot(w.T, x) + b)
    cost = -np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) / m
    dw = np.dot(x, (y_hat - y).T) / m
    db = np.sum(y_hat - y) / m
    assert (dw.shape == w.shape)
    cost = np.squeeze(cost)
    assert (cost.shape == ())
    return {'dw': dw, 'db': db}, float(cost)


def optimize(w, b, x, y, num_iterations, learning_rate, print_cost=False):
    costs = []
    dw, db = 0, 0
    for i in range(1, num_iterations + 1):
        grads, cost = propagate(w, b, x, y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print('Cost after iteration: %d is %f ' % (i, cost))
    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}
    return params, grads, costs


def predict(w: np.ndarray, b, x: np.ndarray):
    m = x.shape[1]
    w = w.reshape(x.shape[0], 1)
    y_hat = sigmoid(np.dot(w.T, x) + b)

    for i in range(y_hat.shape[1]):
        y_hat[0, i] = 1 if y_hat[0, i] > 0.5 else 0

    assert (y_hat.shape == (1, m))
    return y_hat
