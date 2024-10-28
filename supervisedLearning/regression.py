import math
import numpy as np

class Regression(object):
    '''
    Regression model. 

    Parameters:
    -------------
        n_iterations: int
            The number of training iterations the algorithm with tune the weights for.
        learning_rate: float
            The step length that will be used when updating the weights.
    '''
    def __init__(self, n_iterations, learning_rate):
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate

    def fit(self, X, y):
        pass
