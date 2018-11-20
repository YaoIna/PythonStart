import numpy as np
import sys
import cv2
import matplotlib.pyplot as plt

image = cv2.imread('/Users/xxy/Desktop/WallPaper/gamersky_05origin_09_2016728172444E.jpg', 0)
image = cv2.resize(image, (10, 10))
plt.imshow(image, cmap='gray')
plt.show()

img1 = cv2.imread('/Users/xxy/Desktop/WallPaper/gamersky_05origin_09_2016728172444E.jpg', 0)
binaryImage = np.array(img1, copy=True)
newImage = np.where(np.logical_and(-sys.maxsize - 1 <= binaryImage, binaryImage <= 123), 0, binaryImage)
newImage = np.where(np.logical_and(124 <= newImage, newImage <= sys.maxsize), 255, newImage)
plt.imshow(newImage, cmap='gray')
plt.show()
