import numpy as np


class LinearRegression:
    """
     Logistic Regression.

     Parameters
     -----------
     _x_train: array
             input training samples
     _y_train: array
             expected o/p
     _y_pred : list
             predicted values
     _w      : numpy array
             weight matrix for n number of features
     _rssquare : list
             list of Residual Sum of squares
     _lr    : float, default = 0.00001
             learning rate
     _n_iter : int, default = 100
              number of iterations for optimization
     """

    def __init__(self):
        self._x = []
        self._y = []
        self._y_pred = []
        self._w = np.array([])
        self._rssquare = []
        self._lr = 0.00001
        self._n_iter = 100

    """
    Gradient descent function to optimize weight parameters
    For n iterations :
        - calculate residual sum of square error
        - update weight = weight - learning_rate * loss
        - generate new hypothesis
    """

    def _gradient_iterative(self, _lr: float, _n_iter: int):
        for i in range(self._n_iter):
            _rssquare = self._rss()
            self._w = np.array(self._w - (self._lr * _rssquare))
            self._hypothesis()
            self._rssquare = np.append(self._rssquare, _rssquare)

    """
    Hypothesis function to calculate predicted value
    y_predicted = w0*x0 + w1*x1 + w2*x2 + ....
    """

    def _hypothesis(self):
        self._y_pred = np.dot(self._x, np.transpose(self._w))

    """
    Residual Sum of squares errors
    returns error value
    """

    def _rss(self):
        return np.array([np.sum(np.square((self._y, self._y_pred)))], dtype='double')

    """
    Function to train and infer
    _x          : x training samples
    _y          : y training samples
    lr          : default: 1e-3, optional parameter
    n_iter     : default: 100, optional parameter
    
    Examples
    -------------------------------------------------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from mllab.LogisticRegression import LogisticRegression
    >>> from mllab.preprocessing import normalize as nm 
    >>> X, y
    >>> linreg = LinearRegression()
    >>> linreg.train(X, y, lr=1e-5, n_iter=100)
    >>> weight_parameter = linreg.infer()
    """

    def train(self, x, y, lr=1e-3, n_iter=100):
        self._x = x
        self._y = y
        self._lr = lr
        self._n_iter = n_iter
        n_rows, n_columns = self._x.shape
        self._w = np.array([np.random.random(n_columns)])
        self._hypothesis()
        self._gradient_iterative(self._lr, self._n_iter)

    """
    return weight parameter for regression
    """

    def infer(self):
        return self._w
