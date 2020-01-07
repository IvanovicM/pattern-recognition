import numpy as np
from .LinearClassifier import LinearClassifier
from .Classifier import Data

class QuadraticClassifier(LinearClassifier):

    def fit(self, data, method='desired_output'):
        '''
            Fits the model for given training data.
            Args:
                X (numpy array): Data
                Y (numpy array ): Classes for the given data
                method (string): Always 'desired_output'
        '''
        X = self._create_quadratic_features(data['X'])
        new_data = Data(X, data['y'], None, None, None, None)
        super().fit(new_data, method='desired_output')

    def predict(self, X):
        '''
            Predicts output for the given data.
            Args:
                X (numpy array): Data
        '''
        return super().predict(self._create_quadratic_features(X))
    
    def _create_quadratic_features(self, X_lin):
        # Create placeholder
        features_num = X_lin.shape[1]
        quadratic_num = features_num * features_num - 1
        X = np.zeros((X_lin.shape[0], features_num + quadratic_num))

        # Fill with linear features
        for i in range(features_num):
             X[:, i] = X_lin[:, i]

        # Create quadratic features
        curr_quadratic = 2
        for i in range(features_num):
            X[:, curr_quadratic] = np.multiply(X_lin[:, i], X_lin[:, i])
            curr_quadratic += 1

        # Create crosswise quadratic features
        for i in range(features_num):
            for j in np.arange(i + 1, features_num):
                X[:, curr_quadratic] = np.multiply(X_lin[:, i], X_lin[:, j])
                curr_quadratic += 1

        return X
