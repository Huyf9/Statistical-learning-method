import numpy as np


x = np.array([[3, 3],
              [4, 3],
              [1, 1]])
y = np.array([1, 1, -1])
w = np.zeros(2)
b = 0
lr = 1

for i in range(3):
    if y[i]*(np.dot(x[i, :], w.T) + b)<=0:
        w[0] += lr*y[i]*x[i, 0]
        w[1] += lr*y[i]*x[i, 1]
        b += lr*y[i]
