"""
Function to Normalize datasets: interms of columns values
x_norm = (x - mean(x))/standard_deviation(x)
"""


def normalize(x):
    x_avg = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_norm = (x - x_avg) / x_std
    return x_norm


def feature_scaling():
    pass
