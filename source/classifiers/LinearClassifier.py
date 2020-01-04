import numpy as np

class LinerClassifier():

    def fit(self, data, Y, method='resubstitution'):
        '''
            Fits the model for given training data.
            Args:
                X (numpy array of doubles): Data
                Y (numpy array of doubles): Classes for the given data
                method (string): 'iterative' or 'resubstitution'
        '''
        # RRR
        for s = 0 : 0.01 : 1


    def predict(self, X):
        '''
            Predicts output for the given data.
            Args:
                x (numpy array of doubles): Data
        '''
        Y = np.zeros(X.shape[0])