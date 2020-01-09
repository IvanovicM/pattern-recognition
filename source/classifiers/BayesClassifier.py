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

    def estimate_errors(self, xmin=-5, xmax=5, ymin=-5, ymax=5, dx=0.1, dy=0.1):
        '''
            Estimates error ot Bayes classifier.
            Args:
                xmin (double): Minimal value for the first coordinate.
                xmax (double): Maximal value for the first coordinate.
                ymin (double): Minimal value for the second coordinate.
                ymax (double): Maximal value for the second coordinate.
                dx (double): Step for the first coordinate.  
                dy (double): Step for the second coordinate.  
        '''
        if self.f1 is None or self.f2 is None or self.compare_value is None:
            return None
        self.error_1 = 0.0
        self.error_2 = 0.0

        # Estimate error
        for x in np.arange(xmin, xmax, dx):
            for y in np.arange(ymin, ymax, dy):
                f1_value = self.f1([x, y])
                f2_value = self.f2([x, y])
                h = -np.log(f1_value /f2_value)

                # Errors for this classification
                if h < self.compare_value:
                    self.error_2 += f2_value * dx * dy
                else:
                    self.error_1 += f1_value * dx * dy
        return self.error_1, self.error_2
