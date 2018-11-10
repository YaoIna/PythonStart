from sklearn import datasets
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import accuracy_score

dig = datasets.load_digits()
images = dig.images
# images.shape()
plt.imshow(images[0], cmap=plt.cm.gray_r)
plt.show()

x, y = dig.data, dig.target
# x.shape()
# y.shape()

spilt = int(x.shape[0] * 0.7)
train_x = x[:spilt, :]
test_x = x[spilt:, :]
train_y = y[:spilt]
test_y = y[spilt:]

clf = svm.SVC(gamma=.001, C=1)
clf.fit(train_x, train_y)

num = 8
plt.imshow(images[spilt + num], cmap=plt.cm.gray_r)
plt.show()

m = clf.predict(test_x[num].reshape(1, -1))

accuracy_score(train_y, clf.predict(train_x))
accuracy_score(test_y, clf.predict(test_x))
