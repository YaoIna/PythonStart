import numpy


def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def image2vector(x: numpy.ndarray):
    return x.reshape(numpy.multiply.reduce(x.shape), 1)


def normalize_rows(x: numpy.ndarray):
    normal_x = numpy.linalg.norm(x, axis=1, keepdims=True)
    return x / normal_x


def softmax(x: numpy.ndarray):
    x = numpy.exp(x)
    x_sum = numpy.sum(x, axis=1).reshape(2, 1)
    return x / x_sum


def l1(y_hat: numpy.ndarray, y: numpy.ndarray):
    return numpy.sum(numpy.abs(y - y_hat))


def l2(y_hat: numpy.ndarray, y: numpy.ndarray):
    return numpy.sum(numpy.power(y - y_hat, 2))


if __name__ == '__main__':
    param = numpy.asarray([1, 2, 3])
    print(sigmoid(param))
    print(sigmoid_derivative(param))

    image = numpy.array([[[0.67826139, 0.29380381],
                          [0.90714982, 0.52835647],
                          [0.4215251, 0.45017551]],

                         [[0.92814219, 0.96677647],
                          [0.85304703, 0.52351845],
                          [0.19981397, 0.27417313]],

                         [[0.60659855, 0.00533165],
                          [0.10820313, 0.49978937],
                          [0.34144279, 0.94630077]]])

    print(image2vector(image))

    normal_param = numpy.asarray([[0, 3, 4], [1, 6, 4]])
    print(normalize_rows(normal_param))
    softmax_param = numpy.asarray([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])
    print(softmax(softmax_param))

    label_hat = numpy.array([.9, 0.2, 0.1, .4, .9])
    label = numpy.array([1, 0, 0, 1, 1])

    print(l2(label_hat, label))
