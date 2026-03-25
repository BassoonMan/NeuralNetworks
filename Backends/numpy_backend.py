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
    
    def fused_bias_act(self, x, bias, act_type):
        """Fused bias-add + activation: activation(X + B)."""
        xd = np.asarray(x, dtype=np.float32)
        bd = np.asarray(bias, dtype=np.float32)
        if act_type < 0:
            return None
        val = xd + bd
        if act_type == 1:        # leakyReLU
            val = np.where(val >= 0, val, 0.01 * val)
        elif act_type == 2:      # tanh
            val = np.tanh(val)
        return val
    
    def fused_sgd_update(self, weights, velocity, gradient, lr, momentum, decay_factor):
        """In-place: v = mom*v + g; w = decay*w + lr*v"""
        velocity[:] = momentum * velocity + gradient
        weights[:] = decay_factor * weights + lr * velocity

    def fused_sgd_bias_update(self, bias, velocity, gradient, lr, momentum):
        """In-place: v = mom*v + g; b = b + lr*v"""
        velocity[:] = momentum * velocity + gradient
        bias[:] = bias + lr * velocity

    def matmul_at_b_clip(self, a, b, clip):
        """Compute soft_clip(A^T @ B, clip)."""
        ad = np.asarray(a, dtype=np.float32)
        bd = np.asarray(b, dtype=np.float32)
        result = ad.T @ bd
        if clip > 0:
            result = clip * np.tanh(result / clip)
        return result

    def sum_axis0_clip(self, x, clip):
        """Sum along axis 0 with soft clipping."""
        result = np.sum(np.asarray(x, dtype=np.float32), axis=0)
        if clip > 0:
            result = clip * np.tanh(result / clip)
        return result
    
    def delta_act_clip(self, inputs, outputs, act_type, clip):
        """delta_out = soft_clip(input_delta * act_deriv(preact), clip).
        
        inputs: input deltas, outputs: pre-activation cache."""
        inp = np.asarray(inputs, dtype=np.float32)
        pre = np.asarray(outputs, dtype=np.float32)
        if act_type == 1:        # leakyReLU derivative
            d = np.where(pre >= 0, 1.0, 0.01).astype(np.float32)
        elif act_type == 2:      # tanh derivative
            t = np.tanh(pre)
            d = (1.0 - t * t).astype(np.float32)
        else:                    # identity derivative
            d = np.ones_like(pre, dtype=np.float32)
        val = inp * d
        if clip > 0:
            val = clip * np.tanh(val / clip)
        return val
    
    def add3_clip(self, A, B, C, clip):
        """Compute soft_clip(A + B + C, clip)."""
        val = (np.asarray(A, dtype=np.float32)
             + np.asarray(B, dtype=np.float32)
             + np.asarray(C, dtype=np.float32))
        if clip > 0:
            val = clip * np.tanh(val / clip)
        return val

    def permute_split(self, x, perm):
        """Permute columns by perm, then split into two halves."""
        x = np.asarray(x, dtype=np.float32)
        perm = np.asarray(perm, dtype=np.int32)
        permuted = x[:, perm]
        half = permuted.shape[1] // 2
        return permuted[:, :half].copy(), permuted[:, half:].copy()
    
    def concat_invperm(self, x1, x2, inv_perm):
        """Concatenate two halves and apply inverse permutation."""
        x1 = np.asarray(x1, dtype=np.float32)
        x2 = np.asarray(x2, dtype=np.float32)
        inv_perm = np.asarray(inv_perm, dtype=np.int32)
        combined = np.concatenate([x1, x2], axis=1)
        return combined[:, inv_perm].copy()

    def bias_act_dual(self, x, bias, act_type):
        """Return (pre_act, post_act) where pre_act = X+B, post_act = activation(X+B)."""
        x = np.asarray(x, dtype=np.float32)
        bias = np.asarray(bias, dtype=np.float32)
        pre_act = x + bias
        if act_type == 1:        # leakyReLU
            post_act = np.where(pre_act >= 0, pre_act, 0.01 * pre_act)
        elif act_type == 2:      # tanh
            post_act = np.tanh(pre_act)
        else:                    # identity
            post_act = pre_act.copy()
        return pre_act, post_act