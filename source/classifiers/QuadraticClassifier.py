import numpy as np
import math
from .LinearClassifier import LinearClassifier

class QuadraticClassifier(LinearClassifier):

    def fit(self, data, method='resubstitution'):
        '''
            Fits the model for given training data.
            Args:
                X (numpy array): Data
                Y (numpy array ): Classes for the given data
                method (string): 'resubstitution' or 'desired_output'
        '''
        if method == 'resubstitution':
            self._fit_resubstitution(data)
        if method == 'desired_output':
            self._fit_desired_output(data)
