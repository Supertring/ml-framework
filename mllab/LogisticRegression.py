import numpy as np
from .activation import sigmoid
from .loss import binary_cross_entropy
from .optimization import *
from .optimization import optimizer


class LogisticRegression:
    """
    Logistic Regression.

    Parameters
    -----------
    _x_train: ndarray
            input training samples
    _y_train: ndarray
            expected o/p
    _x_test : ndarray
            input testing samples
    _y_test : ndarray
            expected o/p
    _w      : numpy array
            weight matrix for n number of features
    _w_history: numpy array
            history of updated weights with optimization algorithm
    _loss_history: list
            loss values obtained in each iteration of optimization
    _lr     : float, default: 1e-3
            learning rate, determines the step size at each iteration of optimization
    _epoches: int, default: 100
            number of iterations for optimization
    _w_0    : int, default: 0
            bias value used in hypothesis
    _batch_size : int, default = 1
            number of samples used in one iteration
    _prediction : list
            predicted values
    _optimizer : string, default: gradient_descent
            optimization algorithm used for optimization
    _loss_func : string, default: binary_cross_entropy
            loss function to calculate loss
    _activation_func : string, default: sigmoid
            activation function
    """

    def __init__(self):
        self._x_train = []
        self._y_train = []
        self._x_test = []
        self._y_test = []
        self._w = np.array([])
        self._w_history = np.array([])
        self._loss_history = []
        self._lr = 1e-3
        self._epoches = 100
        self._w_0 = 0
        self._batch_size = 1
        self._prediction = []
        self._optimizer = 'gradient_descent'
        self._loss_func = 'binary_cross_entropy'
        self._activation_func = 'sigmoid'

    """
    Function to Initialize optimization algorithm, loss function and activation function.
    If not initialize, object of LogisticRegression will used default parameters
    """

    def compile(self, _optimizer, _loss_func, _activation_func):
        self._optimizer = _optimizer
        self._loss_func = _loss_func
        self._activation_func = _activation_func

    """
    Function to train algorithm
    _x          : x training samples
    _y          : y training samples
    weight      : default: np.array([]), optional parameter
    bias        : default: 0, optional parameter
    lr          : default: 1e-3, optional parameter
    batch_size  : default: 1, optional parameter
    epoches     : default: 100, optional parameter
    
    returns     : _w, optimized weights
                : _loss_history, loss history
    Examples
    -------------------------------------------------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from supertring.LogisticRegression import LogisticRegression
    >>> from supertring.preprocessing import normalize as nm 
    >>> X_train, X_test, y_train, y_test
    >>> logreg = LogisticRegression()
    >>> logreg.compile('gradient_descent','binary_cross_entropy','sigmoid')
    >>> weight, loss_history = logreg.train(_x, _y, weights = np.array([np.zeros(int)]), 
                                            bias=1, lr=1e-5, batch_size=100, epoches=1000)
    >>> predictions = logreg.infer(y_train, y_test)
    """

    def train(self, _x, _y, weights=np.array([]), bias=0, lr=1e-3, batch_size=1, epoches=100):
        self._x_train = _x
        self._y_train = _y
        self._w_0 = bias
        self._lr = lr
        self._batch_size = batch_size
        self._epoches = epoches
        n_rows, n_columns = self._x_train.shape
        if weights.size == 0:
            self._w = np.array([np.zeros(n_columns)])
        else:
            self._w = weights
        self._w, self._w_history, self._loss_history = optimizer(self._optimizer, self._x_train,
                                                                 self._y_train, self._w_0, self._w, self._lr,
                                                                 self._batch_size, self._epoches,
                                                                 self._activation_func, self._loss_func)
        return self._w, self._loss_history

    """
    Function to test
    returns : _prediction, predicted values
    """

    def infer(self, _x, _y):
        self._x_test = _x
        self._y_test = _y
        self._prediction = sigmoid(np.dot(self._x_test, np.transpose(self._w)), self._w_0)
        self._prediction = np.round(self._prediction)
        return self._prediction
