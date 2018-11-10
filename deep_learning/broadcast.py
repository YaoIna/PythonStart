import numpy as np

food = np.asarray([[56, 0, 4.4, 68], [1.2, 104, 52, 8], [1.8, 135, 99, .9]])
food_sum = food.sum(axis=0)
print(food_sum)
# broadcasting
percent = food / food_sum.reshape(1, 4) * 100
print(percent)

# a,b列数相等，broadcast自动复制补全行数
a = np.asarray([[1, 2, 3], [4, 5, 6]])
b = np.asarray([100, 200, 300])
print(a + b)
# a,c行数相等，broadcast自动复制补全列数
c = np.asarray([[100], [200]])
print(a + c)

# 以上适应于+ - * /运算
