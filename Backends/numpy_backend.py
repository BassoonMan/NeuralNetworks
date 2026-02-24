import numpy as np


class NumpyBackend:
    name = "cpu"

    def is_device_array(self, x):
        return isinstance(x, np.ndarray)

    def to_device(self, x, dtype=np.float32):
        return np.asarray(x, dtype=dtype)

    def to_host(self, x):
        return np.asarray(x)

    def matmul(self, a, b):
        return np.matmul(a, b)

    def add(self, a, b):
        return np.add(a, b)

    def subtract(self, a, b):
        return np.subtract(a, b)

    def multiply(self, a, b):
        return np.multiply(a, b)

    def divide(self, a, b):
        return np.divide(a, b)

    def apply_activation(self, x, func_name):
        if func_name == "leakyReLU":
            return np.where(x < 0, 0.01 * x, x)
        if func_name == "tanh":
            return np.tanh(x)
        if func_name == "identity":
            return x
        return None

    def apply_activation_derivative(self, x, func_name):
        if func_name == "leakyReLU":
            return np.where(x < 0, 0.01, 1.0).astype(np.float32)
        if func_name == "tanh":
            t = np.tanh(x)
            return (1.0 - t * t).astype(np.float32)
        if func_name == "identity":
            return np.ones_like(x, dtype=np.float32)
        return None
