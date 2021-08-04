import numpy as np
import random

class Random_Regressor:
    def __init__(self):
        self.weights = None

    def learn(self, x, y) -> None:
        """
        x are the data points
        y are the result that corrispond with x
        """
        self.weights = np.random.rand(x.shape[1])

    def predict(self, x)
    """
    x is the test points
    """
    result = np.dot(x, self.weights)
    return result

class Mean:
    def __init__(self):
        self.mean = None

    def learn(self, x, y):
        self.mean = np.mean(y)

    def predict(self, x):
        result = np.ones(x.shape[0]) * self.mean
        return result

class Simple_Linear_Regression(Random_Regressor):
    """
    simple stochastic gradient based regression
    """
    def __init__(self, epochs: int, batch_size: int,  stepsize_approach = 'heuristic'):
        """
        epoch is epoch size
        batch_size is the number of items per batch
        steosuze approach could either be heuristic or adagrad

        """
        self.epochs = epochs
        self.stepsize_approach = stepsize_approach
        self.batch_size = batch_size

        self.weights = None
        self.num_samples = 0
        self.num_features = 0

    def learn(self, x, y):
        self.num_samples = x.shape[0]
        self.num_features = x.shape[1]

        self.weights = np.random.rand(self.num_features)
        self.g_bar = np.zeros(self.num_features)
        if self.stepsize_approach == "heuristic":
            self.g_bar = 1

        
        # for each epoch
        for p in range(1, self.epochs):
            # TODO: shuffle?

            # for each batch
            for k in range(self.num_samples // self.batch_size):
                gradient_sum = np.zeros(self.num_features)
                c = 0
                
                # get g
                for i in range(k * self.batch_size, min((k+1) * self.batch_size, self.num_samples)):
                    l = np.subtract( np.dot(np.transpose(X[i]), self.weights), y[i] )
                    gradient_sum += np.dot(l, X[i])
                    c += 1

                gradient_sum /= c

                if self.stepsize_approach == "adagrad":
                    for i in range(1, self.num_features):
                        self.g_bar[i] += gradient_sum[i] ** 2
                        self.weights[i] -= gradient_sum[i]/((self.g_bar[i]) ** 0.5)

                else:
                    self.weights -= gradient_sum * self.getStepSize(gradient_sum)
                    # self.weights = self.weights-np.dot(self.getStepSize(gradient_sum/self.batch_size), gradient_sum)

    def getStepSize(self, gradient):
        """
        returns step size based on the approach chosen for heuristic
        """
        if self.stepsize_approach == "heuristic":
            k = 0
            for i in range(self.num_features):
                k += abs(gradient[i])

            self.g_bar += k / (self.num_features + 1)

            # print(self.g_bar)
            return 1 / (self.g_bar+1)
        
    def predict(self, x):
        """
        dot product
        """
        return np.dot(x, self.weights)
