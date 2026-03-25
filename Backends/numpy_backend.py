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
    
    def sum(self, x, axis=None, keepdims=False):
        return np.sum(x, axis=axis, keepdims=keepdims)
    
    def coupling_forward(self, u, s_raw, t, limit=2.0):
        """v = u * exp(soft_clip(s_raw, limit)) + t"""
        s_clipped = limit * np.tanh(np.asarray(s_raw, dtype=np.float32) / limit)
        return np.asarray(u) * np.exp(s_clipped) + np.asarray(t)

    def coupling_backward(self, v, s_raw, t, limit=2.0):
        """u = (v - t) * exp(-soft_clip(s_raw, limit))"""
        s_clipped = limit * np.tanh(np.asarray(s_raw, dtype=np.float32) / limit)
        return (np.asarray(v) - np.asarray(t)) * np.exp(-s_clipped)

    def coupling_s_outer_grad(self, diff, u, s_raw, s_limit=2.0, clip_limit=0.0):
        """outer = diff * u * exp(soft_clip(s, s_limit)) * soft_clip_deriv(s, s_limit)"""
        s = np.asarray(s_raw, dtype=np.float32)
        th = np.tanh(s / s_limit)
        s_clipped = s_limit * th
        clip_deriv = 1.0 - th * th
        result = np.asarray(diff) * np.asarray(u) * np.exp(s_clipped) * clip_deriv
        if clip_limit > 0:
            result = clip_limit * np.tanh(result / clip_limit)
        return result

    def coupling_input_grad(self, diff, s_raw, limit=2.0):
        """du = diff * exp(soft_clip(s_raw, limit))"""
        s_clipped = limit * np.tanh(np.asarray(s_raw, dtype=np.float32) / limit)
        return np.asarray(diff) * np.exp(s_clipped)

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

    def transpose(self, x):
        """Return transposed array."""
        return np.ascontiguousarray(np.asarray(x).T)
