import numpy as np
from deep_learning.gradient_checking.test_case import *
from deep_learning.gradient_checking.gc_utils import sigmoid, relu, dictionary_to_vector, vector_to_dictionary, \
    gradients_to_vector


# 1-dimensional gradient checking
def forward_propagation(x, theta):
    return np.dot(theta, x)


def back_propagation(x):
    return x


def gradient_check(x, theta, epsilon=1e-7):
    theta_plus = theta + epsilon
    theta_minus = theta - epsilon

    j_plus = forward_propagation(x, theta_plus)
    j_minus = forward_propagation(x, theta_minus)

    grad_approx = (j_plus - j_minus) / (2 * epsilon)
    grad = back_propagation(x)

    difference = np.linalg.norm(grad_approx - grad) / (np.linalg.norm(grad_approx) + np.linalg.norm(grad))

    if difference < 1e-7:
        print("The gradient is correct!")
    else:
        print("The gradient is wrong!")

    return difference


def test_one_dimensional():
    x, theta = 2, 4
    difference = gradient_check(x, theta)
    print("difference = " + str(difference))


def forward_propagation_n(x, y, parameters):
    # retrieve parameters
    m = x.shape[1]
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
    Z1 = np.dot(W1, x) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = relu(Z2)
    Z3 = np.dot(W3, A2) + b3
    A3 = sigmoid(Z3)

    # Cost
    log_probs = np.multiply(-np.log(A3), y) + np.multiply(-np.log(1 - A3), 1 - y)
    cost = 1. / m * np.sum(log_probs)
    cache = (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3)
    return cost, cache


def backward_propagation_n(x, y, cache):
    m = x.shape[1]
    (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

    dZ3 = A3 - y
    dW3 = 1. / m * np.dot(dZ3, A2.T)
    db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)

    dA2 = np.dot(W3.T, dZ3)
    dZ2 = np.multiply(dA2, np.int64(A2 > 0))
    dW2 = 1. / m * np.dot(dZ2, A1.T)
    db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

    dA1 = np.dot(W2.T, dZ2)
    dZ1 = np.multiply(dA1, np.int64(A1 > 0))
    dW1 = 1. / m * np.dot(dZ1, x.T)
    db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

    return {"dZ3": dZ3, "dW3": dW3, "db3": db3,
            "dA2": dA2, "dZ2": dZ2, "dW2": dW2, "db2": db2,
            "dA1": dA1, "dZ1": dZ1, "dW1": dW1, "db1": db1}


def gradient_check_n(parameters, gradients, x, y, epsilon=1e-7):
    vector_params, _ = dictionary_to_vector(parameters)
    num_params = vector_params.shape[0]
    j_plus = np.zeros(shape=(num_params, 1))
    j_minus = np.zeros(shape=(num_params, 1))
    for i in range(num_params):
        vector_plus = np.copy(vector_params)
        vector_plus[i, 0] = vector_plus[i, 0] + epsilon
        vector_minus = np.copy(vector_params)
        vector_minus[i, 0] = vector_minus[i, 0] - epsilon
        params_plus = vector_to_dictionary(vector_plus)
        params_minus = vector_to_dictionary(vector_minus)
        cost_plus, _ = forward_propagation_n(x, y, params_plus)
        cost_minus, _ = forward_propagation_n(x, y, params_minus)
        j_plus[i, 0] = cost_plus
        j_minus[i, 0] = cost_minus

    grad_approx = (j_plus - j_minus) / (2 * epsilon)
    grad = gradients_to_vector(gradients)
    difference = np.linalg.norm(grad_approx - grad) / (np.linalg.norm(grad_approx) + np.linalg.norm(grad))

    if difference > 1e-7:
        print(
            "\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference


def test_n_dimensional():
    x, y, parameters = gradient_check_n_test_case()

    cost, cache = forward_propagation_n(x, y, parameters)
    gradients = backward_propagation_n(x, y, cache)
    gradient_check_n(parameters, gradients, x, y)


test_n_dimensional()
