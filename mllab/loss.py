import numpy as np

"""
Function to return loss function as per initialized value
name    : string, name of loss function
y_train : list, expected o/p values
epsilon : default (float = 1e-5), optional
"""


def loss_function(name, y_train, h, epsilon: float = 1e-5):
    if name == 'binary_cross_entropy':
        return binary_cross_entropy(y_train, h, epsilon)
    elif name == 'MSE':
        return MSE()
    elif name == 'MAE':
        return MAE()
    else:
        print('No Match on loss function')


"""
Function to return binary cross entropy loss value
y_train : list, expected o/p values
h       : hypothesis function
binary_cross_entropy = -(y_i * log(h(x_i))) + (1-y_i)* log(1-h(x_i))
"""


def binary_cross_entropy(y_train, h, epsilon: float = 1e-5):
    y1 = np.dot(np.transpose(y_train), np.log(h + epsilon))
    y0 = np.dot((1 - np.transpose(y_train)), (np.log(1 - h + epsilon)))
    return -y1 + y0


def residual_square_mean(y, y_bar,):
    return np.array([np.sum(np.square((y, y_bar)))], dtype='double')


def MSE():
    pass


def MAE():
    pass
