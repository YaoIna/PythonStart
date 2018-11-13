import numpy as np
from deep_learning.deep_neural_image.dnn_app_utils import *


def l_layer_model(x: np.ndarray, y: np.ndarray, layers_dims: list, learning_rate=0.0075, num_iterations=3000,
                  print_cost=False):
    np.random.seed(1)
    params = initialize_parameters_deep(layers_dims)
    costs = []
    for i in range(num_iterations):
        AL, caches = L_model_forward(x, params)
        cost = compute_cost(AL, y)
        grads = L_model_backward(AL, y, caches)
        parameters = update_parameters(params, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return params
