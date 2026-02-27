import numpy as np
class LeakyReLU():
    def __init__(self, scale=0.01):
        self.scale = scale


    def activation(self, x):
        return np.maximum(self.scale * x, x)
    

    def derivative(self, x):
        return np.where(x > 0, 1, self.scale)


class Sigmoid():
    def __init__(self, scale=1):
        self.scale = scale


    def activation(self, x):
        return 1 / (1 + np.exp(-x * self.scale))
    

    def derivative(self, x):
        return self.activation(x) * (1 - self.activation(x))