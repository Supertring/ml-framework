import numpy as np
import pandas as pd

"""
FUNCTION NAME : euclidean_distances
Args    :   Matrix a 
            Matrix b
            
Task    :   Computes euclidean distances between two matrix a and b

Returns :   C (numpy.ndarray), distance matrix
"""


def euclidean_distances(a, b):
    A = np.array(a)
    B = np.array(b)
    A_square = np.reshape(np.sum(A * A, axis=1), (A.shape[0], 1))
    B_square = np.reshape(np.sum(B * B, axis=1), (1, B.shape[0]))
    AB = A @ B.T
    C = -2 * AB + B_square + A_square
    return np.sqrt(C)


"""
FUNCTION NAME : entropy
Args    :   y , List with n number of value

Task    :   Calculates the entropy for given dataset
            summation  of (p(x)*log2(p(x))
            where p(x) = number of count of each distinct value in y / total number of value in y
                EG: y = [1,1,2,2,3,3,4,4]
                THEN, p(x=1) = 2/8
Returns :   entropy : float
"""


def entropy(y):
    # values : list of unique values from y, count : list of number of count for each unique value
    values, count = np.unique(y, return_counts=True)
    # Sum of total number of values in list y
    total_count = np.sum(count)
    # Initially set entropy=0
    _entropy = 0
    # Loop into count : calculate entropy with respect to each unique value in y
    for i in range(len(count)):
        # Calculate entropy and sum it up to _entropy in each loop
        _entropy += (-count[i] / total_count) * (np.log2(count[i] / total_count))
    # return entropy
    return _entropy


"""                                                                                                                    
FUNCTION NAME : information_gain                                                                                                
Args    :    data : Matrix of whole datasets including target values                                                                            
             split_feature : name of the feature upon which information gain has to be calculated
             target_name : name of the target feature                
Task    :   Calculates the information gain for given dataset                                                           
                                                                                              
Returns :   ig : float                                                                                            
"""


def information_gain(data, split_feature, target=None):
    # Total entropy of the target datasets
    total_entropy = entropy(data[target])
    # Get the values and find the number of each values for the split feature
    values, counts = np.unique(data[split_feature], return_counts=True)
    # Total number of values in counts list
    total_count = np.sum(counts)
    # Find the sum of entropy of each unique variable
    sum_entropy = np.sum(
        [(counts[i] / total_count) * entropy(data.where(data[split_feature] == values[i]).dropna()[target])
         for i in range(len(values))])
    # calculate information gain
    ig = total_entropy - sum_entropy
    return ig
