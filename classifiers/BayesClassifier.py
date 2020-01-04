import numpy as np

class BayesClassifier():

    def __init__(self, n_classes):
        self.n_classes = n_classes

    def fit(self, X, y, C):
        '''
            Fits the model for given training data.
            Args:
                X (numpy array of doubles): Data
                Y (numpy array of doubles): Classes for the given data
        '''
        n_examples = X.shape[0]

        # Priors
        self.priors = np.bincount(Y) / n_examples

    def predict(self, X):
        '''
            Predicts output for the given data.
            Args:
                x (numpy array of doubles): Data
        '''
        Y = np.zeros(X.shape[0])