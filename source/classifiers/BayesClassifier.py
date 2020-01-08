import numpy as np
from .Classifier import Classifier

class BayesClassifier(Classifier):

    def __init__(self, f1=None, f2=None, priors=np.array([0.5, 0.5])):
        '''
            Constructor.
            Args:
                f1 (function): PDF for the first class
                f2 (function): PDF for the second class
                priors (numpy array of doubles): Priors for every class.
        '''
        self.f1 = f1
        self.f2 = f2
        self.priors = priors
        self.compare_value = np.log(self.priors[0] / self.priors[1])

    def predict_classes(self, X):
        '''
            Predicts output for the given data.
            Args:
                X (numpy array): Data
        '''
        if self.f1 is None or self.f2 is None or self.compare_value is None:
            return None
        if len(X.shape) == 1:
            X = np.array([X])
        
        # Discriminant function h
        self.h = np.zeros(X.shape[0])
        for i in range(len(self.h)):
            self.h[i] = -np.log(self.f1(X[i, :]) / self.f2(X[i, :]))

        # Predictions
        Y = np.zeros(X.shape[0])
        for i in range(len(Y)):
            if self.h[i] < self.compare_value:
                Y[i] = 0
            else:
                Y[i] = 1
        return Y
