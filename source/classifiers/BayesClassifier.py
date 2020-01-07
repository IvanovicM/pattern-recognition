import numpy as np
from .Classifier import Classifier

class BayesClassifier(Classifier):

    def fit(self, data):
        '''
            Fits the model for given training data.
            Args:
                data (class Data): Data
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