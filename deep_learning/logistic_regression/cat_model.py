import numpy as np

from deep_learning.python_basics_with_numpy import sigmoid


class CatModel:
    def __init__(self):
        self.w = np.asarray([])
        self.b = 0

    def __standardize_data(self, data: np.ndarray):
        data = data.reshape(data.shape[0], -1).T
        return data / 255

    def __initialize_with_zeros(self, dim: int):
        w = np.zeros(shape=(dim, 1))
        b = 0

        assert (w.shape == (dim, 1))
        assert (isinstance(b, float) or isinstance(b, int))
        return w, b

    def __propagate(self, w: np.ndarray, b, x: np.ndarray, y: np.ndarray):
        m = x.shape[1]
        y_hat = sigmoid(np.dot(w.T, x) + b)
        cost = -np.sum((y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))) / m
        dw = np.dot(x, (y_hat - y).T) / m
        db = np.sum(y_hat - y) / m
        assert (dw.shape == w.shape)
        cost = np.squeeze(cost)
        assert (cost.shape == ())
        return {'dw': dw, 'db': db}, float(cost)

    def __optimize(self, w, b, x, y, num_iterations, learning_rate, print_cost=False):
        costs = []
        dw, db = 0, 0
        for i in range(1, num_iterations + 1):
            grads, cost = self.__propagate(w, b, x, y)
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

    def fit(self, train_x: np.ndarray, train_y: np.ndarray, num_iteration, learning_rate, print_cost=False):
        train_x = self.__standardize_data(train_x)
        w, b = self.__initialize_with_zeros(train_x.shape[0])
        params, _, _ = self.__optimize(w, b, train_x, train_y, num_iteration, learning_rate, print_cost)
        self.w = params['w']
        self.b = params['b']

    def predict(self, x: np.ndarray):
        x = self.__standardize_data(x)
        m = x.shape[1]
        self.w = self.w.reshape(x.shape[0], 1)
        y_hat = sigmoid(np.dot(self.w.T, x) + self.b)

        for i in range(y_hat.shape[1]):
            y_hat[0, i] = 1 if y_hat[0, i] > 0.5 else 0

        assert (y_hat.shape == (1, m))
        return y_hat
