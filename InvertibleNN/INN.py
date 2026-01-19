import numpy as np

class InvertibleNeuralNetwork:
    def __init__(self):
        pass
    
    def forward(self, x):
        # Split x into two parts
        # u1, u2 = split(x)
        u1, u2 = np.split(x, 2)
        # v1 = u1 odot exp(s2(u2)) + t2(u2)
        pass

    def backward(self, y):
        pass
