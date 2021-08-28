import numpy as np
from .activation import sigmoid
from .loss import loss_function, binary_cross_entropy
from .activation import activation_function, sigmoid

"""
Function to return optimization function as per initialized value
name    : string, optimization algorithm name
_x_train: training samples
_y_train: exptected o/p samples
_w_0    : default: 0, bias
_w      : initial weight matrix
_lr     : default:1e-3 , learning rate

"""


def optimizer(name, _x_train, _y_train, _w_0, _w, _lr, _batch_size, _epoches, activation_func, loss_func):
    if name == 'gradient_descent':
        return gradient_descent(_x_train, _y_train, _w_0, _w, _lr, _batch_size, _epoches, activation_func, loss_func)
    elif name == 'RMSProp':
        return RMSProp()
    elif name == 'Adagrad':
        return Adagrad()
    elif name == 'Adam':
        return Adam()
    else:
        print("No match on optimizer")


"""
Optimization Algorithm: gradient descent
_x_train    : input features
_y_train    : expected o/p
_w_0        : default:0, bias, optional
_w          : default: np.array([np.zeros(int)]), weight matrix, optional
_lr         : default: 1e-3, learning rate, optional
_batch_size : default: 1, number of samples used in one iteration, optional
_epoches    : default: 100, number of iterations for optimization, optional
_activation_func : default: sigmoid, activation function, optional
_loss_func       : default: binary_cross_entropy, loss function, optional
"""


def gradient_descent(_x_train, _y_train, _w_0, _w, _lr, _batch_size, _epoches, activation_func='', loss_func=''):
    _w_history = np.array([])
    _weights = _w
    _loss_history = []
    for i in range(_epoches):
        x_t = np.transpose(_x_train)
        w_t = np.transpose(_weights)
        # calculate hypothesis
        _h = activation_function(activation_func, _x_train, w_t, _w_0)
        # update weight values
        _weights = np.transpose(np.transpose(_weights) - (_lr / _batch_size) * np.dot(x_t, (_h - _y_train)))
        _w_history = np.append(_w_history, _weights)
        # calculate loss value
        loss_value = loss_function(loss_func, _y_train, _h, 0.0001)
        _loss_history = np.append(_loss_history, loss_value)
    return _weights, _w_history, _loss_history


def RMSProp():
    pass


def Adagrad():
    pass


def Adam():
    pass
