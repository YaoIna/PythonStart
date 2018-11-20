from deep_learning.optimization.opt_utils import initialize_parameters, forward_propagation, compute_cost, \
    backward_propagation, load_dataset, predict, plot_decision_boundary, predict_dec
from deep_learning.optimization.test_case import *
import numpy as np
import matplotlib.pyplot as plt


def update_parameters_with_gd(parameters: dict, grad: dict, learning_rate: float):
    for i in range(int(len(parameters) / 2)):
        parameters['W%d' % (i + 1)] = parameters['W%d' % (i + 1)] - learning_rate * grad['dW%d' % (i + 1)]
        parameters['b%d' % (i + 1)] = parameters['b%d' % (i + 1)] - learning_rate * grad['db%d' % (i + 1)]
    return parameters


def random_mini_batches(x: np.ndarray, y: np.ndarray, mini_batch_size=64, seed=0):
    np.random.seed(seed)
    m = x.shape[1]
    mini_batch_list = []
    permutation = list(np.random.permutation(m))
    shuffled_x = x[:, permutation]
    shuffled_y = y[:, permutation].reshape(1, m)

    num_full_batch = int(np.floor(m / mini_batch_size))
    for i in range(num_full_batch):
        mini_batch_x = shuffled_x[:, mini_batch_size * i:(i + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[:, mini_batch_size * i:(i + 1) * mini_batch_size]
        mini_batch_list.append((mini_batch_x, mini_batch_y))

    if m % mini_batch_size != 0:
        last_batch_start = num_full_batch * mini_batch_size
        last_batch_x = shuffled_x[:, last_batch_start::]
        last_batch_y = shuffled_y[:, last_batch_start::]
        mini_batch_list.append((last_batch_x, last_batch_y))

    return mini_batch_list


def initialize_velocity(parameters: dict):
    velocity = {}
    for k, v in parameters.items():
        velocity['d' + k] = np.zeros_like(v)
        velocity['d' + k] = np.zeros_like(v)
    return velocity


def update_parameters_with_momentum(parameters: dict, grads: dict, velocity: dict, beta: float, learning_rate: float):
    for i in range(len(parameters) // 2):
        velocity['dW%d' % (i + 1)] = beta * velocity['dW%d' % (i + 1)] + (1 - beta) * grads['dW%d' % (i + 1)]
        parameters['W%d' % (i + 1)] = parameters['W%d' % (i + 1)] + learning_rate * velocity['dW%d' % (i + 1)]
        velocity['db%d' % (i + 1)] = beta * velocity['db%d' % (i + 1)] + (1 - beta) * grads['db%d' % (i + 1)]
        parameters['b%d' % (i + 1)] = parameters['b%d' % (i + 1)] + learning_rate * velocity['db%d' % (i + 1)]
    return parameters, velocity


def initialize_adam(parameters: dict):
    adam_dict = {}
    for k, v in parameters.items():
        adam_dict['V' + k] = np.zeros_like(v)
        adam_dict['S' + k] = np.zeros_like(v)
    return adam_dict


def update_parameters_with_adam(parameters: dict, grads: dict, adam: dict, t: int, learning_rate=0.01,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    adam_corrected = {}
    for i in range(len(parameters) // 2):
        adam['VW' + str(i + 1)] = beta1 * adam['VW' + str(i + 1)] + (1 - beta1) * grads['dW' + str(i + 1)]
        adam_corrected['VW' + str(i + 1)] = adam['VW' + str(i + 1)] / (1 - np.power(beta1, t))
        adam['Vb' + str(i + 1)] = beta1 * adam['Vb' + str(i + 1)] + (1 - beta1) * grads['db' + str(i + 1)]
        adam_corrected['Vb' + str(i + 1)] = adam['Vb' + str(i + 1)] / (1 - np.power(beta1, t))

        adam['SW' + str(i + 1)] = beta2 * adam['SW' + str(i + 1)] + (1 - beta2) * np.power(grads['dW' + str(i + 1)], 2)
        adam_corrected['SW' + str(i + 1)] = adam['SW' + str(i + 1)] / (1 - np.power(beta2, t))
        adam['Sb' + str(i + 1)] = beta2 * adam['Sb' + str(i + 1)] + (1 - beta2) * np.power(grads['db' + str(i + 1)], 2)
        adam_corrected['Sb' + str(i + 1)] = adam['Sb' + str(i + 1)] / (1 - np.power(beta2, t))

        parameters['W' + str(i + 1)] = parameters['W' + str(i + 1)] - learning_rate * adam_corrected[
            'VW' + str(i + 1)] / (
                                               np.sqrt(adam_corrected['SW' + str(i + 1)] + epsilon) + epsilon)
        parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - learning_rate * adam_corrected[
            'Vb' + str(i + 1)] / (
                                               np.sqrt(adam_corrected['Sb' + str(i + 1)]) + epsilon)

    return parameters, adam


def model(x, y, layers_dims, optimizer, learning_rate=0.0007, mini_batch_size=64, beta=0.9,
          beta1=0.9, beta2=0.999, epsilon=1e-8, num_epochs=10000, print_cost=True):
    """
    3-layer neural network model which can be run in different optimizer modes.

    Arguments:
    X -- input data, of shape (2, number of examples)
    Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
    layers_dims -- python list, containing the size of each layer
    learning_rate -- the learning rate, scalar.
    mini_batch_size -- the size of a mini batch
    beta -- Momentum hyperparameter
    beta1 -- Exponential decay hyperparameter for the past gradients estimates
    beta2 -- Exponential decay hyperparameter for the past squared gradients estimates
    epsilon -- hyperparameter preventing division by zero in Adam updates
    num_epochs -- number of epochs
    print_cost -- True to print the cost every 1000 epochs

    Returns:
    parameters -- python dictionary containing your updated parameters
    """

    L = len(layers_dims)  # number of layers in the neural networks
    costs = []  # to keep track of the cost
    t = 0  # initializing the counter required for Adam update
    seed = 10  # For grading purposes, so that your "random" minibatches are the same as ours

    # Initialize parameters
    parameters = initialize_parameters(layers_dims)

    # Initialize the optimizer
    if optimizer == "gd":
        pass  # no initialization required for gradient descent
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v, s = initialize_adam(parameters)
        adam = {}
        for i in range(len(v)):
            adam['VW' + str(i + 1)] = v['dW' + str(i + 1)]
            adam['Vb' + str(i + 1)] = v['db' + str(i + 1)]
            adam['SW' + str(i + 1)] = s['dW' + str(i + 1)]
            adam['Sb' + str(i + 1)] = s['db' + str(i + 1)]

    # Optimization loop
    for i in range(num_epochs):

        # Define the random minibatches. We increment the seed to reshuffle differently the dataset after each epoch
        seed = seed + 1
        minibatches = random_mini_batches(x, y, mini_batch_size, seed)

        for minibatch in minibatches:

            # Select a minibatch
            (minibatch_X, minibatch_Y) = minibatch

            # Forward propagation
            a3, caches = forward_propagation(minibatch_X, parameters)

            # Compute cost
            cost = compute_cost(a3, minibatch_Y)

            # Backward propagation
            grads = backward_propagation(minibatch_X, minibatch_Y, caches)

            # Update parameters
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters, grads, learning_rate)
            elif optimizer == "momentum":
                parameters, v = update_parameters_with_momentum(parameters, grads, v, beta, learning_rate)
            elif optimizer == "adam":
                t = t + 1  # Adam counter
                parameters, v = update_parameters_with_adam(parameters, grads, adam,
                                                            t, learning_rate, beta1, beta2, epsilon)

        # Print the cost every 1000 epoch
        if print_cost and i % 1000 == 0:
            print("Cost after epoch %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


train_x, train_y = load_dataset()

# # train 3-layer model
# layers_dims = [train_x.shape[0], 5, 2, 1]
# parameters = model(train_x, train_y, layers_dims, optimizer="gd")
#
# # Predict
# predictions = predict(train_x, train_y, parameters)
#
# # Plot decision boundary
# plt.title("Model with Gradient Descent optimization")
# axes = plt.gca()
# axes.set_xlim([-1.5, 2.5])
# axes.set_ylim([-1, 1.5])
# plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)

# train 3-layer model
layers_dims = [train_x.shape[0], 5, 2, 1]
parameters = model(train_x, train_y, layers_dims, beta=0.9, optimizer="momentum")

# Predict
predictions = predict(train_x, train_y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5, 2.5])
axes.set_ylim([-1, 1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_x, train_y)
