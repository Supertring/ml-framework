import numpy as np
import re
import math

"""
FUNCTION NAME: tokenize_str
Args:   x  : text input
Task :  Tokenize the text 
Returns: list of tokens for input text 
"""


def tokenize_str(text):
    return re.findall(r'\b\w\w+\b', text)


"""
FUNCTION NAME: build_vocab
Args:   x (np array) : input data points
        y (np array) : expected output values 
                                     
Task :  Build a vocabulary 
Returns:(data_info, total_document_count, vocab)
        data_info    : dict,
                       Eg:. { 0: {'doc_count' : 480,
                             'tokens_count' : 14552
                             'tokens' : {'date' : 33, 'from' : 23}
                            }}
                        0: data label/ y (expected value)
                        doc_count : number of document that belongs to label 0
                        tokens_count : number of tokens the belongs to label 0
                        tokens : all the tokens and their number of occurrences in the label 0
        total_document_count   : int
                        Total number of documents in the whole datasets
        vocab    : dict
                        dictionary of all the vocab present in the datasets including their number of occurences
                        Eg:. {'date' : 20, 'from' : 23} 
"""


def build_vocab(x, y):
    # Dictionary to contain document count, tokens count and tokens in each category/class of dataset
    data_info = {}
    # Unique labels/category/expected output
    label = np.unique(y)
    # Total documents in whole input datasets
    total_document_count = 0
    # Real vocabulary containing term and its count for the whole datasets
    vocab = {}
    # Loop into labels and initialized data_info with class/category
    for i in range(len(label)):
        data_info[i] = {"doc_count": 0,
                        "tokens_count": 0,
                        "tokens": {}}
    # Loop into input x,y
    for i in range(len(x)):
        label = y[i]
        # Tokenize the input text
        tokens = tokenize_str(x[i])
        # Increment document count
        total_document_count += 1
        # Increment document count for each category/label/class
        data_info[label]["doc_count"] += 1

        # Loop into tokens
        for token in tokens:
            # Increment token count
            data_info[label]["tokens_count"] += 1
            # Set token into vocab
            if not token in vocab:
                vocab[token] = 1
            else:
                vocab[token] += 1
            # Set token into data_info with specified category/label/class
            if not token in data_info[label]["tokens"]:
                data_info[label]["tokens"][token] = 1
            else:
                data_info[label]["tokens"][token] += 1
    # Final magic, return data_info, total_document_count, vocab
    return data_info, total_document_count, vocab
