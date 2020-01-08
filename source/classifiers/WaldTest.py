import numpy as np

class WaldTest():

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

    def predict_class(self, X, eps1, eps2):
        '''
            Predicts class for the given data.
            Args:
                X (numpy array): Data
                eps1 (float): error for the prediction of the first class
                eps2 (float): error for the prediction of the second class
        '''
        if self.f1 is None or self.f2 is None or self.compare_value is None:
            return None
        if len(X.shape) == 1:
            X = np.array([X])
        
        # Caulculate values a and b
        self.a = -np.log((1- eps1) / eps2)
        self.b = -np.log(eps1 / (1- eps2))
        
        # Calculate s_m
        self.m = 0
        self.s_m = []
        s_m = 0
        for i in range(X.shape[0]):
            h = -np.log(self.f1(X[i, :]) / self.f2(X[i, :]))

            self.m += 1
            s_m += h
            self.s_m.append(s_m)

            # Is the test over?
            if s_m <= self.a:
                return 0
            if s_m >= self.b:
                return 1

        # What if there was no decision?
        return None

    def __getitem__(self, key):
        if key == 'm':
            return self.m
        if key == 's_m':
            return self.s_m
        if key == 'a':
            return self.a
        if key == 'b':
            return self.b
        return None