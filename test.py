import math

# import numpy
#
# d = [1, 2, 3, 4]
#
# w = d[0]
# for i in range(len(d)):
#     d[i] = d[i] + w
# ttt = d.insert(0, 3)
# a = [[1, 3, 4], [5, 5, 8], [0, 7, 4]]
# c = numpy.asarray([[1, 2, 3], [2, 9, 0]])
# b = numpy.asarray(a)
# a = a[1:]
# b = numpy.insert(b, 0, c[0], axis=0)
# b = numpy.insert(b, b.shape[0], c[0], axis=0)
# m = b[1:3]
# tt = 1
from datetime import datetime

import numpy

# d = numpy.asarray([[1, 3, 5], [2, 3, 6], [6, 5, 7]])
# ttt = d.shape
# mm = numpy.stack(d, axis=1)
# tt = mm.shape
# mm = mm.T
# t = 1
# import numpy
#
# a = numpy.asarray([[1, 2, 3], [6, 7, 8], [5, 3, 9]])
# mm = a.shape
# a = a.reshape(a.shape[0], -1) / 10
# tt = a.shape
# mm = 1

# tt = numpy.asarray([[1, 2, 4, 5], [2, 3, 5, 6], [7, 7, 7, 7]])
# numpy.save('aaa.npy', tt)
# mm = numpy.load('aaa.npy')
# ii = 1


# mm = numpy.asarray([1, 2, 3, 5, 6])
# tt = [0, 1, 2, 3]
# mm = mm[tt]
# nn = 1

# tt = [1, 2, 3, 4]
# m = int(numpy.min(tt))
# t = 1


# mm = numpy.asarray([[1, 2, 4], [2, 3, 5]])
# tt = numpy.square(mm)
# ttt = 1

tt = numpy.random.permutation(6)
mm = 1

# def probability_winning_series(n: int, p: float) -> float:
#     win_number = int(numpy.ceil(n / 2)) + 1
#     p_array = numpy.zeros(shape=(win_number + 1, win_number + 1))
#     p_array[0, :] = 1
#     p_array[0, 0] = -1
#     for i in range(1, win_number):
#         for j in range(1, win_number):
#             p_array[i, j] = p * p_array[i - 1, j] + (1 - p) * p_array[i, j - 1]
#     return p_array[win_number, win_number]
#
#
# probability_winning_series(7, 0.4)
