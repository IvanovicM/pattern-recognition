import numpy as np
import math

class Classifier():

    def predict(self, X):
        '''
            Predicts output for the given data.
            Args:
                X (numpy array): Data
        '''
        return None

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

    def get_prediction_line(self, X, eps=0.1):
        # Meshgrid data
        x1_min = min(X[:, 0])
        x1_max = max(X[:, 0])
        num1 = math.ceil((x1_max - x1_min) / 0.1)
        x1 = np.linspace(x1_min, x1_max, num=num1)
        
        x2_min = min(X[:, 1])
        x2_max = max(X[:, 1])
        num2 = math.ceil((x2_max - x2_min) / 0.1)
        x2 = np.linspace(x2_min, x2_max, num=num2)

        # Generate data for prediction
        values = np.zeros((len(x1) * len(x2), 2))
        values[:, 0] = np.repeat(x1, len(x2))
        values[:, 1] = np.tile(x2, len(x1))

        # Get predicted values close to zero
        y = self.predict(values)
        if y is None:
            return None, None
        idx = [i for i in range(len(y)) if abs(y[i]) < eps]

        x1_zero_close = [values[i, 0] for i in idx]
        x2_zero_close = [values[i, 1] for i in idx]
        return x1_zero_close, x2_zero_close

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