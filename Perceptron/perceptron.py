import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt


def dataitem(data_len):
    x1_data = np.array([[2*(i+1)+3+np.random.normal(0, 2) for i in range(data_len)],
                        [3*(i+1)+1+np.random.normal(0, 2) for i in range(data_len)]]).T
    x1_label = np.ones((data_len, 1))

    x2_data = np.array([[0.7*(i+1)+2+np.random.normal(0, 2) for i in range(data_len)],
                        [0.6*(i+1)+1+np.random.normal(0, 2) for i in range(data_len)]]).T
    x2_label = -np.ones((data_len, 1))

    x_data = np.concatenate((x1_data, x2_data), axis=0)
    x_label = np.concatenate((x1_label, x2_label), axis=0)

    return np.concatenate((x_data, x_label), axis=1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def loss(x_data, pred):
    return (x_data[2] - pred)**2 / 2

def optim(w, b, lr, x, pred):
    w[0] += (x[2]-pred)*x[0]*lr
    w[1] += (x[2]-pred)*x[1]*lr
    b += (x[2]-pred)*lr
    return w, b

def cal_pred(w, b, x_data):
    return w[0]*x_data[0] + w[1]*x_data[1] + b

if __name__ == '__main__':
    w = np.array([0.1, -0.3])
    b = 0
    lr = 0.001
    x_data = dataitem(200)
    drawdata(x_data, 400)

    for epoch in range(100):
        total_loss_value = 0
        nums_train = 10
        for i in range(200):
            output = sigmoid(cal_pred(w, b, x_data[i]))
            loss_value = loss(x_data[i], output)
            total_loss_value += loss_value
            w, b = optim(w, b, lr, x_data[i], output)
        print("第{}轮的Loss是：".format(epoch+1), total_loss_value/nums_train)
