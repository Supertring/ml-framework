import pandas as pd
import numpy as np


class SimpleNN:
    """
     KMeans.

     Parameters
     -----------
     _x              : array
                        input training points, size N:D
     _y              : array
                        expected output
     _activation     : default : "tanh"
                        activation function name, that defines the output of a node
     _input_dim      : int
                        number of neural node in input layer
     _output_dim     : int
                        number of neural node in output layer
     _epoches        : int : default : 10
                        number of iterations for optimization
     _print_loss     : boolean : default : False
                        to check if loss is to be printed or not
     _w1             : list
                        weight matrix for layer1
     _w2             : list
                        weight matrix for layer2
     _w3             : list
                        weight matrix for layer3
     _b1             : list
                        bias matrix for layer1
     _b2             : list
                        bias matrix for layer2
     _b3             : list
                        bias matrix for layer3
     _reg_value      : float
                        regularization value,used for tuning the function. Controls the excessively fluctuating function
     _epsilon        : float
                        learning rate, determines the step size at each iteration of optimization
     _model          : dictionary
                        dictionary of learned parameters, weight and bias
     _batch_size     : int
                        number of samples used in one iteration
     _batch_loc      : list
                        contains the location for all batch of dataset

     """

    def __init__(self):
        self._x = []
        self._y = []
        self._activation = "tanh"
        self._input_dim = 0
        self._hidden_dim = 0
        self._output_dim = 0
        self._epoches = 10
        self._print_loss = False
        self._w1 = []
        self._w2 = []
        self._w3 = []
        self._w4 = []
        self._w5 = []
        self._w6 = []
        self._w7 = []
        self._w8 = []

        self._b1 = []
        self._b2 = []
        self._b3 = []
        self._b4 = []
        self._b5 = []
        self._b6 = []
        self._b7 = []
        self._b8 = []

        self._reg_value = 0.01
        self._epsilon = 0.01
        self._model = {}
        self._batch_size = 32
        self._batch_loc = []

    """
    FUNCTION NAME : tanh, (derivative form of tanh for backward propagation)
    Args    :   value , float

    Task    :   Calculate derivative of tanh function  : f'(x) =. 1 - f(x)^2   
    Returns :   float value
    """

    def tanh(value):
        return (1 - np.power(value, 2))

    """
    FUNCTION NAME : relu, (derivative form of relu for backward propagation)
    Args    :   value , float

    Task    :   Calculate derivative of relu function  : f'(x) = 0 if x<0, 1 if x>=0
    Returns :   float value
    """

    def relu(value):
        return max(0.0, value.all())

    """
    FUNCTION NAME : forward_prop, 
    Args    :   x (array), input value
            :   w1, w2, w3 (list), weight parameters
            :   b1, b2, b3 (list), bias parameters

    Task    :   Do forward propagation: activation_function(wTx + b)  
    Returns :   probs, probability scores as output
    """
    """
    Not the best way to do..For now weights and bias terms and calculation are done manually step by step
    Need to create a
    """
    def forward_prop(self, x, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, b4, b5, b6, b7, b8):
        # Forward propogation
        # perform wTx + b
        z1 = x.dot(w1) + b1
        # Applies tanh activation function to (wTx +b)
        a1 = np.tanh(z1)

        z2 = a1.dot(w2) + b2
        a2 = np.tanh(z2)

        z3 = a2.dot(w3) + b3
        a3 = np.tanh(z3)

        z4 = a3.dot(w4) + b4
        a4 = np.tanh(z4)

        z5 = a4.dot(w5) + b5
        a5 = np.tanh(z5)

        z6 = a5.dot(w6) + b6
        a6 = np.tanh(z6)

        z7 = a6.dot(w7) + b7
        a7 = np.tanh(z7)

        z8 = a7.dot(w8) + b8
        # Get the exponential value of z3, nice for introducting non linearity, smooth and differential, imp for back prop.
        #make optimization objective convex, which ensure global minimum
        exp_scores = np.exp(z8)
        # Calculate output as probability scores
        probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        return probs

    """
    FUNCTION NAME : loss, 
    Args    :   x (array), input value
            :   y (array), 
            :   model , dict of weight and bias terms
            :   reg_value , regularization parameter
            :   n_samples, input sames from specified batch

    Task    :   calculate loss value
    Returns :   loss
    """

    def loss(self, x, y, model, reg_value, n_samples):

        w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, b4, b5, b6, b7, b8 = model['w1'], model['w2'], model['w3'], model['w4'], \
                                                                         model['w5'], model['w6'], model['w7'], model['w8'], \
                                                                         model['b1'], model['b2'], model['b3'], model['b4'], \
                                                                         model['b5'], model['b6'], model['b7'], model['b8']

        # Forward propagation to calculate predictions
        probs = self.forward_prop(x, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, b4, b5, b6, b7, b8)
        # loss calculations
        #Log  is considered better for two reasons: It prevents "underflow" 
        log_prob = -np.log(probs[range(n_samples), y])
        data_loss = np.sum(log_prob)
        # add regularization term to loss (optional)
        #data_loss += reg_value / 2 * (np.sum(np.square(w1)) + np.sum(np.square(w2)))
        return 1. / n_samples * data_loss

    """
    FUNCTION NAME : batch, 
    Args    :   n_samples (int), total sample of datasets
            :   batch_size(int), size of batch to be created

    Task    :   Create a list that contains location for each batch
    Returns :   batch_loc, locations for all the batch datasets
    """

    def batch(self, n_samples, batch_size):
        batch_loc = []
        q, mod = divmod(n_samples, batch_size)
        a = 0
        b = batch_size
        batch_loc.append([a, b])
        for i in range(q):
            a = b + 1
            b = b + batch_size
            if i + 1 == q:
                b = n_samples
                batch_loc.append([a, b])
            else:
                batch_loc.append([a, b])
        return batch_loc

    """
    FUNCTION NAME : add, 
    Args    :   activation , name of activation function : eg: tanh
            :   input_dim(int), number of neural node in input layer
            :   hidden(int),    number of neural node in hidden layer
            :   output_dim(int), number of neural node in output layer
            :   print_loss (boolean), If true print loss values

    Task    :   Assign values to the parameters of neural nets 
                that creates neural net architecture
    """

    def add(self, activation=tanh, input_dim=0, hidden_dim=0, output_dim=0, print_loss=False):
        self._activation = activation
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._output_dim = output_dim
        self._print_loss = print_loss
        np.random.seed(0)
        self._w1 = np.random.randn(input_dim, hidden_dim) / np.sqrt(input_dim)
        self._b1 = np.zeros((1, hidden_dim))

        self._w2 = np.random.randn(hidden_dim, hidden_dim)
        self._b2 = np.random.randn(1, hidden_dim)

        self._w3 = np.random.randn(hidden_dim, hidden_dim)
        self._b3 = np.random.randn(1, hidden_dim)

        self._w4 = np.random.randn(hidden_dim, hidden_dim)
        self._b4 = np.random.randn(1, hidden_dim)

        self._w5 = np.random.randn(hidden_dim, hidden_dim)
        self._b5 = np.random.randn(1, hidden_dim)

        self._w6 = np.random.randn(hidden_dim, hidden_dim)
        self._b6 = np.random.randn(1, hidden_dim)

        self._w7 = np.random.randn(hidden_dim, hidden_dim)
        self._b7 = np.random.randn(1, hidden_dim)

        self._w8 = np.random.randn(hidden_dim, output_dim)
        self._b8 = np.random.randn(1, output_dim)

    """
    FUNCTION NAME : info, 
    Task    :   Display the neural net architecture that is created.
    """

    def info(self):
        print("Model : Simple Neural Network")
        print("-------------------------------------------------")
        print("Layer \t size \t input weights \t input bais")
        print("-------------------------------------------------")
        print("Input \t {0} ".format(self._input_dim))
        print("-------------------------------------------------")
        print("Hidden \t {0} \t {1} \t {2}".format(self._hidden_dim, self._w1.shape, self._b1.shape))
        print("-------------------------------------------------")
        print("Hidden \t {0} \t {1} \t {2}".format(self._hidden_dim, self._w2.shape, self._b2.shape))
        print("-------------------------------------------------")
        print("Hidden \t {0} \t {1} \t {2}".format(self._hidden_dim, self._w3.shape, self._b3.shape))
        print("-------------------------------------------------")
        print("Hidden \t {0} \t {1} \t {2}".format(self._hidden_dim, self._w4.shape, self._b4.shape))
        print("-------------------------------------------------")
        print("Hidden \t {0} \t {1} \t {2}".format(self._hidden_dim, self._w5.shape, self._b5.shape))
        print("-------------------------------------------------")
        print("Hidden \t {0} \t {1} \t {2}".format(self._hidden_dim, self._w6.shape, self._b6.shape))
        print("-------------------------------------------------")
        print("Hidden \t {0} \t {1} \t {2}".format(self._hidden_dim, self._w7.shape, self._b7.shape))
        print("-------------------------------------------------")
        print("Output \t {0} \t {1} \t {2}".format(self._output_dim, self._w8.shape, self._b8.shape))
        print("-------------------------------------------------")

    """
    FUNCTION NAME : train, 
    Args    :   x (array) , input datasets
            :   y (array),  expected output values
            :   epoches (int), number of iterations for optimization
            :   reg(float), regularization value
            :   epsilon (float), learning rate
            :   batch_size(int), number of samples used in one iteration

    Task    :   Learns from input, output data and creates a learned model
            :   Assigns learned parameters (weight  and bias) to model
    """

    def train(self, x, y, epoches=10, reg_value=0.01, epsilon=0.01, batch_size=32):
        self._x = x
        self._y = y
        self._epoches = epoches
        self._reg_value = reg_value
        self._epsilon = epsilon
        self._batch_size = batch_size

        # generate mini-batch location
        n_samples = len(self._x)
        self._batch_loc = self.batch(n_samples, self._batch_size)

        # get activation function
        activation_function = getattr(SimpleNN, self._activation)

        # Gradient descent for each batch
        for i in range(0, epoches):
            for j in range(len(self._batch_loc)):
                start = self._batch_loc[j][0]
                end = self._batch_loc[j][1]

                # Forward propogation with mini batch
                z1 = self._x[start:end].dot(self._w1) + self._b1
                a1 = np.tanh(z1)

                z2 = a1.dot(self._w2) + self._b2
                a2 = np.tanh(z2)

                z3 = a2.dot(self._w3) + self._b3
                a3 = np.tanh(z3)

                z4 = a3.dot(self._w4) + self._b4
                a4 = np.tanh(z4)

                z5 = a4.dot(self._w5) + self._b5
                a5 = np.tanh(z5)

                z6 = a5.dot(self._w6) + self._b6
                a6 = np.tanh(z6)

                z7 = a6.dot(self._w7) + self._b7
                a7 = np.tanh(z7)

                z8 = a7.dot(self._w8) + self._b8
                exp_scores = np.exp(z8)
                exp_scores_sum = np.sum(exp_scores, axis=1, keepdims=True)
                probs = exp_scores / exp_scores_sum

                # Backward propogation with mini batch
                delta9 = probs
                n_samples = len(self._x[start:end])

                delta9[range(n_samples), y[start:end]] -= 1
                dw8 = (a7.T).dot(delta9)
                db8 = np.sum(delta9, axis=0, keepdims=True)

                delta8 = delta9.dot(self._w8.T) * activation_function(a7)
                dw7 = (a6.T).dot(delta8)
                db7 = np.sum(delta8, axis=0, keepdims=True)

                delta7 = delta8.dot(self._w7.T) * activation_function(a6)
                dw6 = (a5.T).dot(delta7)
                db6 = np.sum(delta7, axis=0, keepdims=True)

                delta6 = delta7.dot(self._w6.T) * activation_function(a5)
                dw5 = (a4.T).dot(delta6)
                db5 = np.sum(delta6, axis=0, keepdims=True)

                delta5 = delta6.dot(self._w5.T) * activation_function(a4)
                dw4 = (a3.T).dot(delta5)
                db4 = np.sum(delta5, axis=0, keepdims=True)

                delta4 = delta5.dot(self._w4.T) * activation_function(a3)
                dw3 = (a2.T).dot(delta4)
                db3 = np.sum(delta4, axis=0, keepdims=True)

                delta3 = delta4.dot(self._w3.T) * activation_function(a2)
                dw2 = (a1.T).dot(delta3)
                db2 = np.sum(delta3, axis=0, keepdims=True)

                delta2 = delta3.dot(self._w2.T) * activation_function(a1)
                dw1 = np.dot(self._x[start:end].T, delta2)
                db1 = np.sum(delta2, axis=0)



                # Add regularization terms (bias do not have regularization terms)
                dw8 += self._reg_value * self._w8
                dw7 += self._reg_value * self._w7
                dw6 += self._reg_value * self._w6
                dw5 += self._reg_value * self._w5
                dw4 += self._reg_value * self._w4
                dw3 += self._reg_value * self._w3
                dw2 += self._reg_value * self._w2
                dw1 += self._reg_value * self._w1


                # Update paramters
                self._w1 += -self._epsilon * dw1
                self._b1 += -self._epsilon * db1
                self._w2 += -self._epsilon * dw2
                self._b2 += -self._epsilon * db2
                self._w3 += -self._epsilon * dw3
                self._b3 += -self._epsilon * db3

                self._w4 += -self._epsilon * dw4
                self._b4 += -self._epsilon * db4
                self._w5 += -self._epsilon * dw5
                self._b5 += -self._epsilon * db5
                self._w6 += -self._epsilon * dw6
                self._b6 += -self._epsilon * db6

                self._w7 += -self._epsilon * dw7
                self._b7 += -self._epsilon * db7
                self._w8 += -self._epsilon * dw8
                self._b8 += -self._epsilon * db8


                # annealing the learning rate (may decrease accuracy in less dataset)
                # if i % 1000 == 0:
                #    self._epsilon = self._epsilon/2
                # Update new parameters to the model
                self._model = {'w1': self._w1, 'b1': self._b1,
                               'w2': self._w2, 'b2': self._b2,
                               'w3': self._w3, 'b3': self._b3,
                               'w4': self._w4, 'b4': self._b4,
                               'w5': self._w5, 'b5': self._b5,
                               'w6': self._w6, 'b6': self._b6,
                               'w7': self._w7, 'b7': self._b7,
                               'w8': self._w8, 'b8': self._b8}

                # display loss
                if self._print_loss and i % 2000 == 0:
                    print("Loss after iteration %i: %f" % (
                        i, self.loss(self._x[start:end], self._y[start:end], self._model, self._reg_value, n_samples)))

    """
    FUNCTION NAME : infer, 
    Args    :   testx (array) , test datasets

    Task    :   Performs forward propagation with the final learned parameters (taken from learned model)

    Returns :   predicted value
    """

    def infer(self, testx):

        # Get all the learned parameters i.e:. weight and bias  from learn model during training
        w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, b4, b5, b6, b7, b8 = self._model['w1'], self._model['w2'], self._model['w3'], self._model['w4'], \
                                                                         self._model['w5'], self._model['w6'], self._model['w7'], self._model['w8'], \
                                                                         self._model['b1'], self._model['b2'], self._model['b3'], self._model['b4'], \
                                                                         self._model['b5'], self._model['b6'], self._model['b7'], self._model['b8']


        # Do Forward propagation, returns probability
        probs = self.forward_prop(testx, w1, w2, w3, w4, w5, w6, w7, w8, b1, b2, b3, b4, b5, b6, b7, b8)
        # get the location of maximum probabilities, that becomes the predicted value
        # eg: [0.9125 0.565], first value in the list is high, so predicted value is 0
        return np.argmax(probs, axis=1)
