import numpy as np

class LinerClassifier():

    def fit(self, data, Y, method='resubstitution'):
        '''
            Fits the model for given training data.
            Args:
                X (numpy array): Data
                Y (numpy array ): Classes for the given data
                method (string): 'iterative' or 'resubstitution'
        '''
        # RRR
        #ßfor s = 0 : 0.01 : 1


    def predict(self, X):
        '''
            Predicts output for the given data.
            Args:
                x (numpy array of doubles): Data
        '''
        Y = np.zeros(X.shape[0])

class Data():

    def __init__(self, X, y, M1, S1, M2, S2):
        self.X = X
        self.y = y
        self.M1 = M1
        self.S1 = S1
        self.M2 = M2
        self.S2 = S2
    
    def __getitem__(self, key):
        if key == 'X':
            return self.X
        if key == 'y':
            return self.y
        if key == 'M1':
            return self.M1
        if key == 'S1':
            return self.S1
        if key == 'M2':
            return self.M2
        if key == 'S2':
            return self.S2
        return None