import numpy as np
import math

class LinearClassifier():

    def fit(self, data, method='resubstitution'):
        '''
            Fits the model for given training data.
            Args:
                X (numpy array): Data
                Y (numpy array ): Classes for the given data
                method (string): 'iterative' or 'resubstitution'
        '''
        # Init optimal values
        self.s = 0
        self.v0 = 0

        # Indeces for both classes
        class0_idx = [i for i in range(len(data['y'])) if data['y'][i] == 0]
        class1_idx = [i for i in range(len(data['y'])) if data['y'][i] == 1]
        error = len(data['y'])

        for s in np.arange(0, 1.01, 0.01):
            # Optimal V for the given s
            V = np.matmul(
                    np.linalg.inv(
                        np.add(
                            np.dot(s, data['S1']),
                            np.dot((1-s), data['S2'])
                        )
                    ),
                    np.subtract(data['M2'], data['M1'])
            )

            # Value Y for every example X[i]
            Y = np.matmul(data['X'], np.transpose(V))
            
            # Find optimal v0
            for v0 in np.arange(math.floor(min(Y)), math.floor(max(Y)) + 1):
                error_0 = len(
                    [Y[class0_idx[i]]
                        for i in range(len(class0_idx))
                            if Y[class0_idx[i]] > v0]
                )
                error_1 = len(
                    [Y[class1_idx[i]]
                        for i in range(len(class1_idx))
                            if Y[class1_idx[i]] < v0]
                )

                if error_0 + error_1 < error:
                    error = error_0 + error_1
                    self.v0 = -v0
                    self.s = s

        # Calculate optimal V
        self.V = np.matmul(
            np.linalg.inv(
                np.add(
                    np.dot(self.s, data['S1']),
                    np.dot((1-self.s), data['S2'])
                )
            ),
            np.subtract(data['M2'], data['M1'])
        )

    def predict(self, X):
        '''
            Predicts output for the given data.
            Args:
                X (numpy array): Data
        '''
        if self.V is None or self.s is None or self.v0 is None:
            return None

        y = np.add(np.matmul(X, np.transpose(self.V)), self.v0)
        return y

    def predict_classes(self, X):
        '''
            Predicts classes for the given data.
            Args:
                X (numpy array): Data
        '''
        y = self.predict(X)
        if y is None:
            return None

        y_classes = [int(y_example > 0) for y_example in y]
        return y_classes

    def prediction_error(self, X, y):
        '''
            Returns prediciton error given data.
            Args:
                X (numpy array): Data
                y (numpy array): True outputs
        '''
        y_predicted = self.predict_classes(X)
        if y_predicted is None:
            return None

        error = len([i for i in range(len(y)) if y_predicted[i] != y[i]])
        return error

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