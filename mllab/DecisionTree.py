import numpy as np
import pandas as pd
from .metrics import entropy
from .metrics import information_gain


class DecisionTree:
    """
     DecisionTree

     Parameters
     -----------
     tree             : dictionary
                        Contains the learned tree by ID3 algorithm
     """

    def __init__(self):
        self.tree = {}

    """
    FUNCTION NAME: train
    Args:   X (array) : input data points
            y (array) : output/expected values  

    Task :  Runs ID3 algorithm, 
            these points will be used as initial centroids  
    """

    def train(self, X, y):
        # Set X to train_test
        train_test = X
        # Get the feature name of input datasets
        features = X.columns
        # Get the target feature name
        target = y.name
        # Combine X and y into train_test dataset
        train_test[target] = y
        # Run ID3 Algorithm
        tree = self.ID3(train_test, train_test, features, target=target, node=None)

    """
    FUNCTION NAME: ID3 
    Args:   train_test (array)          : input data X, including expected output y
            original_train_test (array) : input data X, including expected output y
            features (list)             : list of features of train_test datasets
            target                      : feature name of expected output
            node                        : Used to set the feature value that have maximum number of count

    Task :  Runs ID3 Algorithm

    Returns :    tree (dict)            : learned tree in dict form    
    """

    def ID3(self, train_test, original_train_test, features, target=None, node=None):
        # Check if all the values in y (target output) are same and return the value
        # If unique of y <= 1, means it has only one unique value, ie. y has same values
        # Eg:. if y = [1], i.e. only single unique value
        if len(np.unique(train_test[target])) <= 1:
            # return the actual unique value, eg: return 1
            return np.unique(train_test[target])[0]

        # if train_test (whole dataset) is null,Return the value that have maximum number of count from original dataset
        # Check if length of train_test:input is 0 eg:. train_test =  [], is null
        elif len(train_test) == 0:
            # Set values : unique values of y(target value), counts : number of count of corresponding unique value
            # Eg:. values : array([1, 2, 3, 4, 5, 6, 7]), counts : array([41, 20,  5, 13,  4,  8, 10])
            values, counts = np.unique(original_train_test[target], return_counts=True)
            # Get the value that have maximum number of count,
            # From above Eg:. comment, max_count_value = 1, i.e count = 41
            max_count_value = np.unique(original_train_test[target])[np.argmax(counts)]
            # Return the value that have maximum number of count in the dataset
            return max_count_value

        # Check if feature is empty, and return feature value of parent node.
        elif len(features) == 0:
            return node

        else:
            # Set the initial value for the node : i.e. the target feature value having max count.
            # Set values : unique values of y(target values), counts : number of count of corresponding unique value
            # Eg:. values : array([1, 2, 3, 4, 5, 6, 7]), counts : array([41, 20,  5, 13,  4,  8, 10])
            values, counts = np.unique(train_test[target], return_counts=True)
            # Get the value that have maximum number of count,
            # From above Eg:. comment, max_count_value = 1, i.e count = 41
            max_count_value = np.unique(train_test[target])[np.argmax(counts)]
            # Set the  feature value that have maximum number of count in the dataset to node
            node = max_count_value

            # Select feature that can split the dataset in best way
            # feature_info_gain: list contains information gain for each feature
            features_info_gain = []
            # Loop into each feature and calculate information gain for each feature
            for feature in features:
                # Append the information gain to the features_info_gain list
                features_info_gain.append(information_gain(train_test, feature, target))
            # Get the index of best feature from features_info_gain
            best_feature_index = np.argmax(features_info_gain)
            # Get the best feature from features with index best_feature_index
            best_feature = features[best_feature_index]

            # Set the best_feature to the tree, which is also the root of the tree.
            # with maximum information gain
            tree = {best_feature: {}}

            # Feature that corresponds to highest information gain is removed
            features = [i for i in features if i != best_feature]

            # Expand the tree, below the root node
            for val in np.unique(train_test[best_feature]):
                # Best feature have the largest information gain, so split the dataset as per it
                # Create another datasets without the best feature
                new_train_test = train_test.where(train_test[best_feature] == val).dropna()
                # y = train_test[target_attribute_name]
                # Expand the tree with recursion, using ID3 algorithm
                child_tree = self.ID3(new_train_test, original_train_test, features, target, node)
                # Set child_tree to the existing tree
                tree[best_feature][val] = child_tree
            self.tree = tree
        return self.tree

    """
    FUNCTION NAME: infer
    Args:   testx (array) : test dataset

    Task    : Predicts for unseen data from the learned tree, 
    
    Returns : y_pred (list) : predicted output
    """

    def infer(self, testx):
        # Converts the testx input to dictionary, Note : learned tree is in the form of dictionary
        testx = testx.to_dict(orient='records')
        # List to hold predicted output
        y_pred = []
        # Loop into each test values for prediction
        for i in range(len(testx)):
            # Predicts the output
            pred = self.search_tree(testx[i], self.tree, 1)
            # Appends predicted value to y_pred list
            y_pred.append(pred)
        # Returns predicted value
        return y_pred

    """
    FUNCTION NAME: search_tree
    Args    : values (dict) : test dataset value
            : tree (dict)   : tree learned by the ID3 Algorithm
            : default = 1   : 

    Task    : Predicts for unseen data from the learned tree, 

    Returns : output        : predicted output
    """

    def search_tree(self, values, tree, default=1):
        for k in list(values.keys()):
            # Check the feature k:from values is in tree.keys(): i.e.: root node
            if k in list(tree.keys()):
                # try if the learned tree can classify the unseen data, if not go to except.
                try:
                    output = tree[k][values[k]]
                except:
                    # if the learned tree have no idea of new input value, then return 1,
                    return default
                # we want to work on the node of the tree that is equal to k: (key from values)
                # i.e.: search tree for the node that is below the root node in existing branches.
                # it can be a subtree or leaf node as well.
                output = tree[k][values[k]]
                # here we test weather output is a subtree or a leaf node
                # check if the the output is a leaf node, i.e leaf node is not a dict
                # if it is a dictionary, i.e not a leaf node, recursion happens for search tree
                # else if it is not a dictionary, then it is a leaf node and return the output.
                if isinstance(output, dict):
                    # recursion: is done only in the subtree(output) and not done for whole dataset from root node
                    return self.search_tree(values, output)
                else:
                    return output
