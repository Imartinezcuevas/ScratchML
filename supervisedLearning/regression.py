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

    def initialize_weights(self, n_features):
        """
        Initialize weights randomly
        """
        limit = 1 / math.sqrt(n_features)
        self.w = np.random.uniform(-limit, limit, (n_features, ))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        X = np.insert(X, 0, 1, axis=1)
        self.initialize_weights(n_features=X.shape[1])
        self.training_errors = []

        # Do gradient descent for n_iterations:
        for i in range(self.n_iterations):
            y_pred = np.dot(X, self.w)
            # Calculate loss
            mse = np.mean(0.5 * (y-y_pred)**2 + self.regularization(self.w))
            self.training_errors.append(mse)
            # Calculate gradient
            grad_w = -(y-y_pred).dot(X) + self.regularization.grad(self.w)

            self.w -= self.learning_rate * grad_w

    def predict(self, X):
        X = np.insert(X, 0, 1, axis=1)
        y_pred = X.dot(self.w)
        return y_pred
    
class LinearRegression(Regression):
    """Linear model.
    Parameters:
    -----------
    n_iterations: float
        The number of training iterations the algorithm will tune the weights for.
    learning_rate: float
        The step length that will be used when updating the weights.
    """
    def __init__(self, n_iterations=100, learning_rate=0.001):
        # No regularization
        self.regularization = lambda x: 0
        self.regularization.grad = lambda x: 0
        super(LinearRegression, self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        super(LinearRegression, self).fit(X, y)

class l1_regularization(): # Lasso 
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.linalg.norm(w)
    
    def grad(self, w):
        return self.alpha * np.sign(w)
    
class l2_regularization(): # Ridge
    def __init__(self, alpha):
        self.alpha = alpha

    def __call__(self, w):
        return self.alpha * np.dot(w,w)
    
    def grad(self, w):
        return self.alpha * w
    
class LassoRegression(Regression):
    def __init__(self, reg_factor, n_iterations=100, learning_rate=0.001):
        self.regularization = l1_regularization(alpha=reg_factor)
        super(LassoRegression, self).__init__(n_iterations, learning_rate)

    def fit(self, X, y):
        super(LassoRegression, self).fit(X, y)
