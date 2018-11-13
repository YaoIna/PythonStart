import numpy as np
from deep_learning.deep_neural_image.dnn_app_utils import *


def two_layer_model(x: np.ndarray, y: np.ndarray, layers_dim: tuple, learning_rate=0.0075, num_iterations=3000,
                    print_cost=False):
    np.random.seed(1)
    (n_x, n_h, n_y) = layers_dim

    params = initialize_parameters(n_x, n_h, n_y)
    costs = []

    for i in range(num_iterations):
        W1 = params['W1']
        b1 = params['b1']
        W2 = params['W2']
        b2 = params['b2']

        A1, cache1 = linear_activation_forward(x, W1, b1, 'relu')
        A2, cache2 = linear_activation_forward(A1, W2, b2, 'sigmoid')

        cost = compute_cost(A2, y)

        dA2 = - (np.divide(y, A2) - np.divide(1 - y, 1 - A2))
        dA1, dW2, db2 = linear_activation_backward(dA2, cache2, 'sigmoid')
        _, dW1, db1 = linear_activation_backward(dA1, cache1, 'relu')
        grads = {'dW1': dW1, 'db1': db1, 'dW2': dW2, 'db2': db2}
        params = update_parameters(params, grads, learning_rate=learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration {}: {}".format(i, np.squeeze(cost)))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return params
