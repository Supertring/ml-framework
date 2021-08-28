import numpy as np
import math
import re
from .textprocessing import build_vocab
from .textprocessing import tokenize_str


class NaiveBayes:
    """
     Naive Bayes

     Parameters
     -----------
     x             : np array
                        input training points, size N:D
     y             : np array
                        target/expected values
     data_info     : dict
                        0: data label/ y (exptected value)
                        doc_count : number of document that belongs to label 0
                        tokens_count : number of tokens the belongs to label 0
                        tokens : all the tokens and their number of occurences in the label 0

                    Eg:. { 0: {'doc_count' : 480,
                             'tokens_count' : 14552
                             'tokens' : {'date' : 33, 'from' : 23}
                            }
                          }
     total_document_count   : int
                        Total number of documents in the whole datasets
     vocabulary    : dict
                        dictionary of all the vocab present in the datasets including their number of occurences
                        Eg:. {'date' : 20, 'from' : 23}

     priors        : dict
                        {'0' : 0.235, '1' : 0.568}
                        0 : category/lass
                        0.235 : prior probability value

                        Prior probability for each class/category/label
                        Eg:. log(p(c=0)) = log(number of items in c = 0) - log(total items in whole datasets)
                             log(p(c=11)) = log(480) - log(11314)

     conditionals  : dict
                        {0 : {'date': 0.356,
                               'from' : 0.557}
                        }
                        Conditional probability of each term in input datasets
                        Eg:. conditional probability of a term = log(term count in particular class) - log(token size in a class + size of vocabulary)
                        p(A/B) = p(A intersection B) / p(B)
     """

    def __init__(self):
        self.x = []
        self.y = []
        self.data_info = None
        self.total_document_count = None
        self.vocabulary = {}
        self.priors = {}
        self.conditionals = {}

    """
    FUNCTION NAME: train
    Args:   x (np array) : input data points
            y (np array) : expected output values  

    Task :  Build vocabulary
            Calculate the prior probability for each category/class of datasets, 
            Calculate the conditional probability of each term   
    """

    def train(self, x, y):
        # build vocabulary and returns data_info, total_document_count and vocabulary for input x, y
        self.data_info, self.total_document_count, self.vocabulary = build_vocab(x, y)
        # Loop into each category/class of datasets in data_info
        for category in self.data_info:
            # calculate prior probability for each category/class of data
            self.priors[category] = math.log(self.data_info[category]['doc_count']) - math.log(
                self.total_document_count)
            # Assign category to conditionals probability dict
            self.conditionals[category] = {}
            # Get all the tokens that belongs for specified category/class
            category_tokens = self.data_info[category]['tokens']
            # Get all the tokens values for specified category/class
            tokens_values = category_tokens.values()
            # Get the sum of token size
            category_token_size = 0
            for val in tokens_values:
                category_token_size += val
            # Loop into vocabulary to count term for specified category/class
            for term in self.vocabulary:
                term_count = 1.
                # Sum up term count in each category
                term_count += category_tokens[term] if term in category_tokens else 0.
                # Calculate conditional probability for each term and set to dict : conditionals
                self.conditionals[category][term] = math.log(term_count) - math.log(
                    category_token_size + len(self.vocabulary))

    """
    FUNCTION NAME: predict
    Args:   token_list  : list of tokens for input text

    Task :  Predict the category/class of input text(given as tokens)

    Returns: Category having maximum score 
             Eg:. {'0' : 0.235, '1' : 0.568}, Here return is : 1 
    """

    def predict(self, token_list: list):
        # Dict to set prior probability Eg:. {'0' : 0.235, '1' : 0.568}
        scores = {}
        # Enumerate into data_info
        for _, category in enumerate(self.data_info):
            # Set prior probability data of specified category to score dict
            scores[category] = self.priors[category]
            # Loop into token list
            for term in token_list:
                # If term present in vocabulary
                if term in self.vocabulary:
                    # Add conditional probability of term to scores for specified category/class
                    # scores are assigned with prior probability in the beginning
                    scores[category] += self.conditionals[category][term]
        # Returns category having maximum score
        # Eg:. {'0' : 0.235, '1' : 0.568}, Here return is : 1
        return max(scores, key=scores.get)

    """
    FUNCTION NAME: infer
    Args:   x (np array) : input test data points

    Task :  Predict the category/class of unseen data points
    
    Returns: predicted_y (np array), predicted output  
    """

    def infer(self, x):
        # List to append predicted output
        predicted_y = []
        # Loop into each input fo test input
        for i in range(len(x)):
            # Tokenize the input and create a token list
            token_list = tokenize_str(x[i])
            # Feed token list to predict function that predicts the category/class of your input data
            y_pred = self.predict(token_list)
            # Append predicted output to List
            predicted_y.append(y_pred)
        # Return all predicted output
        return predicted_y
