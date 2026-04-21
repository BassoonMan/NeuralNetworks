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
        elif act_type == 3:
            half = val.shape[1] // 2
            val[:, :half] = np.tanh(val[:, :half])
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
        elif act_type == 3:
            half = pre.shape[1] // 2
            d = np.ones_like(pre, dtype=np.float32)
            t = np.tanh(pre[:, :half])
            d[:, :half] = (1.0 - t * t).astype(np.float32)
        else:                    # identity derivative
            d = np.ones_like(pre, dtype=np.float32)
        val = inp * d
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
        elif act_type == 3:
            half = pre_act.shape[1] // 2
            post_act = pre_act.copy()
            post_act[:, :half] = np.tanh(post_act[:, :half])
        else:                    # identity
            post_act = pre_act.copy()
        return pre_act, post_act
    
    def coupling_forward_merged(self, u, st_raw, limit=2.0):
        """v = u * exp(soft_clip(s_raw, limit)) + t, 
        where s_raw = st_raw[:, :half], t = st_raw[:, half:]"""
        st = np.asarray(st_raw, dtype=np.float32)
        half = st.shape[1] // 2
        s_clipped = limit * np.tanh(st[:, :half] / limit)
        return np.asarray(u) * np.exp(s_clipped) + st[:, half:]
    
    def subtract_divide_fuse(self, a, b, d):
        return (np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) / float(d)
    
    def subtract_divide_loss(self, a, b, d):
        """Fused: out = (A-B)/D, returns (out, sum_of_squares_scalar)."""
        out = (np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) / float(d)
        return out, float(np.sum(out * out))
    
    def fused_coupling_grads_concat(self, s1_outer, diff2, diff1_total, u1, st2_raw,
                                s_limit=2.0, s_clip_limit=1.0, t_clip_limit=0.5):
        """CPU path: build diffs_st1 and diffs_st2 in one call."""
        s1_outer = np.asarray(s1_outer, dtype=np.float32)
        diff2 = np.asarray(diff2, dtype=np.float32)
        diff1_total = np.asarray(diff1_total, dtype=np.float32)
        u1 = np.asarray(u1, dtype=np.float32)
        st2_raw = np.asarray(st2_raw, dtype=np.float32)
        half = s1_outer.shape[1]

        # diffs_st2 left half: coupling_s_outer_grad
        s_raw = st2_raw[:, :half]
        th = np.tanh(s_raw / s_limit)
        s_clipped = s_limit * th
        clip_deriv = 1.0 - th * th
        s2_outer = diff1_total * u1 * np.exp(s_clipped) * clip_deriv
        if s_clip_limit > 0:
            s2_outer = s_clip_limit * np.tanh(s2_outer / s_clip_limit)

        # t portions: soft_clip
        if t_clip_limit > 0:
            t1_clipped = t_clip_limit * np.tanh(diff2 / t_clip_limit)
            t2_clipped = t_clip_limit * np.tanh(diff1_total / t_clip_limit)
        else:
            t1_clipped = diff2.copy()
            t2_clipped = diff1_total.copy()

        diffs_st1 = np.concatenate([s1_outer, t1_clipped], axis=1)
        diffs_st2 = np.concatenate([s2_outer, t2_clipped], axis=1)
        return diffs_st1, diffs_st2
    
    def coupling_s_outer_concat(self, diff, u, st_raw, s_limit=2.0, clip_limit=2.0):
        """Fused s_outer_grad + concat: returns (combined, s_outer)."""
        st = np.asarray(st_raw, dtype=np.float32)
        half = st.shape[1] // 2
        th = np.tanh(st[:, :half] / s_limit)
        s_clipped = s_limit * th
        clip_deriv = 1.0 - th * th
        s_outer = np.asarray(diff) * np.asarray(u) * np.exp(s_clipped) * clip_deriv
        if clip_limit > 0:
            s_outer = clip_limit * np.tanh(s_outer / clip_limit)
        combined = np.concatenate([s_outer, np.asarray(diff, dtype=np.float32)], axis=1)
        return combined, s_outer.copy()
    
    def add2_clip(self, a, b, clip):
        """Compute soft_clip(A + B, clip)."""
        val = np.asarray(a, dtype=np.float32) + np.asarray(b, dtype=np.float32)
        if clip > 0:
            val = clip * np.tanh(val / clip)
        return val
    
    def matmul_bt(self, a, b_transposed):
        """Compute A @ B^T."""
        return np.asarray(a, dtype=np.float32) @ np.asarray(b_transposed, dtype=np.float32).T

    def soft_clip(self, x, limit=2.0):
        """limit * tanh(x / limit)"""
        return limit * np.tanh(np.asarray(x, dtype=np.float32) / limit)

    def fused_forward_invperm_merged(self, v1, u2, inv_perm, st_raw, limit=2.0):
        """Fused: apply coupling to u2 using st_raw, then concat [v1, v2] and inverse-permute."""
        st = np.asarray(st_raw, dtype=np.float32)
        half = st.shape[1] // 2
        s_clipped = limit * np.tanh(st[:, :half] / limit)
        v2 = np.asarray(u2, dtype=np.float32) * np.exp(s_clipped) + st[:, half:]
        combined = np.concatenate([np.asarray(v1, dtype=np.float32), v2], axis=1)
        return combined[:, np.asarray(inv_perm, dtype=np.int32)].copy()

    def coupling_input_grad_merged(self, diff, st_raw, limit=2.0):
        """du = diff * exp(soft_clip(s_raw, limit)), where s_raw = st_raw[:, :half]."""
        st = np.asarray(st_raw, dtype=np.float32)
        half = st.shape[1] // 2
        s_clipped = limit * np.tanh(st[:, :half] / limit)
        return np.asarray(diff, dtype=np.float32) * np.exp(s_clipped)

    def coupling_backward_merged(self, v, st_raw, limit=2.0):
        """u = (v - t) * exp(-soft_clip(s_raw, limit)), where s/t come from st_raw halves."""
        st = np.asarray(st_raw, dtype=np.float32)
        half = st.shape[1] // 2
        s_clipped = limit * np.tanh(st[:, :half] / limit)
        return (np.asarray(v, dtype=np.float32) - st[:, half:]) * np.exp(-s_clipped)

    def coupling_backward_outer_concat(self, diff, u, st_raw, s_limit=2.0, s_clip_limit=1.0, t_clip_limit=0.5):
        """Outer gradient for backward coupling u = (v - t) * exp(-s_clip).
        Returns combined (B, 2*half) where:
          left  = soft_clip(-diff * u * s_clip_deriv, s_clip_limit)
          right = soft_clip(-diff * exp(-s_clip), t_clip_limit)
        """
        diff = np.asarray(diff, dtype=np.float32)
        u = np.asarray(u, dtype=np.float32)
        st = np.asarray(st_raw, dtype=np.float32)
        half = st.shape[1] // 2
        th = np.tanh(st[:, :half] / s_limit)
        s_clipped = s_limit * th
        clip_deriv = 1.0 - th * th
        s_outer = -diff * u * clip_deriv
        if s_clip_limit > 0:
            s_outer = s_clip_limit * np.tanh(s_outer / s_clip_limit)
        t_outer = -diff * np.exp(-s_clipped)
        if t_clip_limit > 0:
            t_outer = t_clip_limit * np.tanh(t_outer / t_clip_limit)
        return np.concatenate([s_outer, t_outer], axis=1)

    def coupling_backward_input_grad_merged(self, diff, st_raw, limit=2.0):
        """dv = diff * exp(-soft_clip(s_raw, limit)), where s_raw = st_raw[:, :half]."""
        st = np.asarray(st_raw, dtype=np.float32)
        half = st.shape[1] // 2
        s_clipped = limit * np.tanh(st[:, :half] / limit)
        return np.asarray(diff, dtype=np.float32) * np.exp(-s_clipped)