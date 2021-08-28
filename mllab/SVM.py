import numpy as np


class SVM:
    """
     Support Vector Machine.

     Parameters
     -----------
     x             : array
                        input training points, size N:D
     y             : int
                        target/expected values
     lmbda         : float
                        regularization parameter
     lr            : float
                        Learning parameter regularized as 1/_lmbda*(_epoches + 1),
                        This parameter decreases as the number of epoches increases.
     n_features    : int
                        number of features for input datasets + addition of bias term : 1
     epoches       : int, default : 100
                        number of iterations for optimization
     w             : list
                        weight matrix for n number of features and bias term (i.e:. (n+1))
     b             : float: default = 0.
                        bias
     """

    def __init__(self, lr=0.001, lmbda=0.01, epoches=1000):
        self.x = []
        self.y = []
        self.lmbda = lmbda
        self.lr = lr
        self.n_features = None
        self.epoches = epoches
        self.w = []
        self.b = 0.

    """    
    Function to predicts    
    Task    :   Multiplication between updated weight matrix and input feature matrix x    
    Returns :   np.dot(x,w) - b  
    """

    def predict(self, x, w, b):
        return np.dot(x, w) - b

    """
    Function : train
    Task     : Train with     
    """

    def train(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        for epo in range(self.epoches):
            """self.lr = 1. / self.lmbda * (epo + 1)"""
            for i, x in enumerate(X):
                if y[i] * self.predict(x, self.w, self.b) < 1:
                    self.w = self.w + (self.lr * (np.dot(x, y[i]) + (-2 * self.lmbda * self.w)))
                    self.b = self.b - self.lr * y[i]
                else:
                    self.w = self.w + (self.lr * (-2 * self.lmbda * self.w))

    def infer(self, x_test):
        pred = self.predict(x_test, self.w, self.b)
        return np.sign(pred)
