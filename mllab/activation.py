import numpy as np

"""
Function to return activation function as per initialized value
activation_func   : string, name of activation function
_x_train          : ndarray, training samples
w_t               : transpose of original weight matrix
_w_0              : bias term
"""
def activation_function(activation_func, _x_train, w_t, _w_0):
    if activation_func == 'sigmoid':
        return sigmoid(np.dot(_x_train, w_t), _w_0)
    elif activation_func == 'ReLu':
        return ReLU()
    elif activation_func == 'Softmax':
        return Softmax()
    elif activation_func == 'Tanh':
        return Tanh()
    else:
        print('No Match on activation function')

"""
returns : 1/(1 + np.exp(-z)) + bias
"""
def sigmoid(z, _w_0):
    return 1 / (1 + np.exp(-z)) + _w_0

def ReLU():
    pass

def Softmax():
    pass

def Tanh():
    pass
