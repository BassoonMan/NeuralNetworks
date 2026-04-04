import numpy as np

try:
    import pyopencl as cl
    import pyopencl.array as cl_array
    import pyopencl.clmath as clmath
except Exception:  # pragma: no cover
    cl = None
    cl_array = None
    clmath = None

try:
    import pyclblast
except Exception:  # pragma: no cover
    pyclblast = None

class _DeviceBufferRing:
    """Round-robin pool of pre-allocated device buffers for a fixed shape.
    Each call to next() returns the next buffer in a circular fashion.
    Safe when the number of simultaneously live results never exceeds pool size.
    """
    __slots__ = ("_bufs", "_idx")

    def __init__(self, queue, shape, n, dtype=np.float32):
        self._bufs = [cl_array.empty(queue, shape, dtype=dtype) for _ in range(n)]
        self._idx = 0

    def next(self):
        buf = self._bufs[self._idx]
        self._idx = (self._idx + 1) % len(self._bufs)
        return buf


class OpenCLBackend:
    # CUDA mental model mapping:
    # - Context            -> cl.Context
    # - Stream             -> cl.CommandQueue
    # - Device tensor      -> pyopencl.array.Array
    # - Kernel launch      -> kernel(queue, global_size, local_size, ...)
    # - cudaMemcpy D2H/H2D -> .get() / cl_array.to_device(...)
    name = "opencl"

    # Naive GEMM kernel (row-major) for float32:
    # C[M, N] = A[M, K] @ B[K, N]
    # One work-item computes one C[row, col].
    _GEMM_SOURCE = r"""
    __kernel void matmul_f32(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const int M,
        const int N,
        const int K)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);

        if (row >= M || col >= N) {
            return;
        }

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }

        C[row * N + col] = acc;
    }

    __kernel void add_row_broadcast_f32(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const int M,
        const int N)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);

        if (row >= M || col >= N) {
            return;
        }

        C[row * N + col] = A[row * N + col] + B[col];
    }
    """

    _FUSED_SOURCE = r"""
    __kernel void soft_clip_f32(
        __global const float* X,
        __global float* Y,
        const float limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        Y[i] = limit * tanh(X[i] / limit);
    }

    __kernel void sum_axis0_f32(
        __global const float* X,
        __global float* Y,
        const int M,
        const int N)
    {
        const int col = get_global_id(0);
        if (col >= N) return;
        float acc = 0.0f;
        for (int row = 0; row < M; ++row) {
            acc += X[row * N + col];
        }
        Y[col] = acc;
    }

    // Backward coupling: u = (v - t) * exp(-soft_clip(s_raw, limit))
    __kernel void coupling_backward_f32(
        __global const float* V,
        __global const float* S_raw,
        __global const float* T,
        __global float* U,
        const float limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        float s_clipped = limit * tanh(S_raw[i] / limit);
        U[i] = (V[i] - T[i]) * exp(-s_clipped);
    }

    // Outer gradient for s-network:
    // outer = diff * u * exp(soft_clip(s_raw, limit)) * (1 - tanh(s_raw/limit)^2)
    __kernel void coupling_s_outer_grad_f32(
        __global const float* diff,
        __global const float* U,
        __global const float* S_raw,
        __global float* outer,
        const float limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        float scaled = S_raw[i] / limit;
        float th = tanh(scaled);
        float s_clipped = limit * th;
        float clip_deriv = 1.0f - th * th;
        outer[i] = diff[i] * U[i] * exp(s_clipped) * clip_deriv;
    }

    // Input gradient: du = diff * exp(soft_clip(s_raw, limit))
    __kernel void coupling_input_grad_f32(
        __global const float* diff,
        __global const float* S_raw,
        __global float* du,
        const float limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        float s_clipped = limit * tanh(S_raw[i] / limit);
        du[i] = diff[i] * exp(s_clipped);
    }

    // Fused soft_clip + optional clamp for gradient outer products:
    // out = clamp(soft_clip(diff * u * exp(soft_clip(s_raw, s_limit)) * clip_deriv, out_limit), -guard, guard)
    __kernel void coupling_s_outer_grad_clipped_f32(
        __global const float* diff,
        __global const float* U,
        __global const float* S_raw,
        __global float* outer,
        const float s_limit,
        const float out_limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        float scaled = S_raw[i] / s_limit;
        float th = tanh(scaled);
        float s_clipped = s_limit * th;
        float clip_deriv = 1.0f - th * th;
        float val = diff[i] * U[i] * exp(s_clipped) * clip_deriv;
        // Apply soft_clip to the output
        outer[i] = out_limit * tanh(val / out_limit);
    }

    __kernel void leaky_relu_f32(
        __global const float* X,
        __global float* Y,
        const float alpha,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        const float x = X[i];
        Y[i] = x >= 0.0f ? x : alpha * x;
    }

    __kernel void leaky_relu_deriv_f32(
        __global const float* X,
        __global float* Y,
        const float alpha,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        Y[i] = X[i] >= 0.0f ? 1.0f : alpha;
    }

    __kernel void tanh_deriv_f32(
        __global const float* X,
        __global float* Y,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        float t = tanh(X[i]);
        Y[i] = 1.0f - t * t;
    }
    // Fused bias-add + activation in a single pass.
    // Y[row, col] = activation(X[row, col] + B[col])
    // act_type: 0 = identity, 1 = leakyReLU (alpha=0.01), 2 = tanh
    __kernel void fused_bias_act_f32(
        __global const float* X,       // (M, N) matmul output
        __global const float* B,       // (1, N) bias row
        __global float* Y,             // (M, N) output
        const int M,
        const int N,
        const int act_type)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= N) return;

        float val = X[row * N + col] + B[col];

        // Apply activation in-place
        if (act_type == 1) {          // leakyReLU
            val = val >= 0.0f ? val : 0.01f * val;
        } else if (act_type == 2) {   // tanh
            val = tanh(val);
        }
        // act_type == 0: identity (no-op)

        Y[row * N + col] = val;
    }
    // Fused SGD + momentum + weight decay update.
    // velocity_new = momentum * velocity + gradient
    // w_new = w * (1 - lr * wd) + lr * velocity_new
    __kernel void fused_sgd_momentum_f32(
        __global float* W,             // weights (read/write)
        __global float* V,             // velocity (read/write)
        __global const float* G,       // gradient
        const float lr,
        const float momentum,
        const float decay_factor,      // pre-computed: 1.0 - lr * wd
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;

        float v = momentum * V[i] + G[i];
        V[i] = v;
        W[i] = decay_factor * W[i] + lr * v;
    }

    // Same for bias (no weight decay typically, but you can add it)
    __kernel void fused_sgd_momentum_bias_f32(
        __global float* B,
        __global float* V,
        __global const float* G,
        const float lr,
        const float momentum,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;

        float v = momentum * V[i] + G[i];
        V[i] = v;
        B[i] = B[i] + lr * v;
    }
    // C[M, N] = soft_clip(A^T[M, K] @ B[K, N], limit)
    // A stored as [K, M], so A^T[m, k] = A[k*M + m]
    __kernel void matmul_at_b_clip_f32(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const int M,
        const int N,
        const int K,
        const float clip_limit)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= N) return;

        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[k * M + row] * B[k * N + col];
        }

        // Fuse soft_clip into the store
        if (clip_limit > 0.0f) {
            acc = clip_limit * tanh(acc / clip_limit);
        }
        C[row * N + col] = acc;
    }
    __kernel void sum_axis0_clip_f32(
    __global const float* X,
    __global float* Y,
    const int M,
    const int N,
    const float clip_limit)
    {
        const int col = get_global_id(0);
        if (col >= N) return;

        float acc = 0.0f;
        for (int row = 0; row < M; ++row) {
            acc += X[row * N + col];
        }

        if (clip_limit > 0.0f) {
            acc = clip_limit * tanh(acc / clip_limit);
        }
        Y[col] = acc;
    }
    // delta_out = soft_clip(input_delta * act_deriv(preact), clip_limit)
    // act_type: 0 = identity, 1 = leakyReLU, 2 = tanh
    __kernel void fused_delta_act_clip_f32(
        __global const float* input_delta,  // (M, N) — proto deltas or diffs
        __global const float* preact,       // (M, N) — pre-activation cache
        __global float* output_delta,       // (M, N) — result
        const int act_type,
        const float clip_limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;

        // Compute activation derivative inline
        float d;
        if (act_type == 1) {           // leakyReLU derivative
            d = preact[i] >= 0.0f ? 1.0f : 0.01f;
        } else if (act_type == 2) {    // tanh derivative
            float t = tanh(preact[i]);
            d = 1.0f - t * t;
        } else {                       // identity derivative
            d = 1.0f;
        }

        float val = input_delta[i] * d;

        // Fuse soft_clip
        if (clip_limit > 0.0f) {
            val = clip_limit * tanh(val / clip_limit);
        }

        output_delta[i] = val;
    }
    // out = soft_clip(A + B + C, limit)
    __kernel void add3_clip_f32(   
        __global const float* A, __global const float* B, __global const float* C,
        __global float* out, const float clip_limit, const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;

        float val = A[i] + B[i] + C[i];

        if (clip_limit > 0.0f) {
            val = clip_limit * tanh(val / clip_limit);
        }

        out[i] = val;
    }
    // Reads x[row, perm[col]], writes to u1/u2 based on whether col < half
    __kernel void permute_split_f32(
    __global const float* X, __global const int* perm,
    __global float* U1, __global float* U2,
    const int M, const int full_N, const int half_N)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= full_N) return;
        float val = X[row * full_N + perm[col]];
        if (col < half_N) {
            U1[row * half_N + col] = val;
        } else {
            U2[row * half_N + (col - half_N)] = val;
        }
    }

    // Forward coupling: v = u * exp(soft_clip(s_raw, limit)) + t
    __kernel void coupling_forward_f32(
        __global const float* U,
        __global const float* S_raw,
        __global const float* T,
        __global float* V,
        const float limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        float s_clipped = limit * tanh(S_raw[i] / limit);
        V[i] = U[i] * exp(s_clipped) + T[i];
    }

    // out[row, inv_perm[col]] = left_or_right_half[row, col]
    __kernel void concat_invperm_f32(
        __global const float* U1, __global const float* U2,
        __global const int* inv_perm, __global float* out,
        const int M, const int half_N)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= 2 * half_N) return;
        int src_col = inv_perm[col];
        float val;
        if (src_col < half_N) {
            val = U1[row * half_N + src_col];
        } else {
            val = U2[row * half_N + (src_col - half_N)];
        }
        out[row * (2 * half_N) + col] = val;
    }

    __kernel void fused_forward_invperm_f32(
        __global const float* V1,
        __global const float* U2,
        __global const int* inv_perm,
        __global const float* S_raw,
        __global const float* T,
        __global float* out,
        const int M,
        const int half_N,
        const float limit)
    {

        float s_clipped;

        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= 2 * half_N) return;
        int src_col = inv_perm[col];
        float val;
        if (src_col < half_N) {
            val = V1[row * half_N + src_col];
        } else {
            s_clipped = limit * tanh(S_raw[row * half_N + (src_col - half_N)] / limit);
            val = U2[row * half_N + (src_col - half_N)] * exp(s_clipped) + T[row * half_N + (src_col - half_N)];
        }
        out[row * (2 * half_N) + col] = val;
    }


    // One kernel, two outputs:
    //   pre_act[row, col] = X[row, col] + B[col]
    //   post_act[row, col] = activation(X[row, col] + B[col])
    __kernel void fused_bias_act_dual_f32(
        __global const float* X,        // (M, N) matmul result
        __global const float* B,        // (1, N) bias
        __global float* pre_act,        // (M, N) output: X + B
        __global float* post_act,       // (M, N) output: activation(X + B)
        const int M,
        const int N,
        const int act_type)             // 0=identity, 1=leakyReLU, 2=tanh
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= N) return;

        float val = X[row * N + col] + B[col];
        pre_act[row * N + col] = val;       // cache for backprop

        if (act_type == 1) {
            val = val >= 0.0f ? val : 0.01f * val;
        } else if (act_type == 2) {
            val = tanh(val);
        }

        post_act[row * N + col] = val;      // activated output
    }

    // Transposes a matrix
    __kernel void transpose_f32(
        __global const float* X,        // (M, N) matrix to transpose
        __global float* out,        // (N, M) Transposed output
        const int M,
        const int N)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= N) return;

        out[col * M + row] = X[row * N + col];
    }
    """

    def __init__(self, batch_size=128, vector_size=64, internal_width=64, preferred_device_type="gpu"):
        # Prefer GPU when available; fallback selection is handled in factory.
        if cl is None or cl_array is None:
            raise RuntimeError("pyopencl is not available")

        platforms = cl.get_platforms()
        if not platforms:
            raise RuntimeError("No OpenCL platforms found")

        device_type = cl.device_type.GPU if preferred_device_type.lower() == "gpu" else cl.device_type.ALL
        selected_device = None
        for platform in platforms:
            devices = platform.get_devices(device_type=device_type)
            if devices:
                selected_device = devices[0]
                break

        if selected_device is None:
            for platform in platforms:
                devices = platform.get_devices()
                if devices:
                    selected_device = devices[0]
                    break

        if selected_device is None:
            raise RuntimeError("No OpenCL devices found")

        self.context = cl.Context([selected_device])
        self.queue = cl.CommandQueue(self.context)
        self._program = cl.Program(self.context, self._GEMM_SOURCE + self._FUSED_SOURCE).build()
        self._matmul_kernel = cl.Kernel(self._program, "matmul_f32")
        self._add_row_broadcast_kernel = cl.Kernel(self._program, "add_row_broadcast_f32")
        self._soft_clip_kernel = cl.Kernel(self._program, "soft_clip_f32")
        self._sum_axis0_kernel = cl.Kernel(self._program, "sum_axis0_f32")
        self._coupling_forward_kernel = cl.Kernel(self._program, "coupling_forward_f32")
        self._coupling_backward_kernel = cl.Kernel(self._program, "coupling_backward_f32")
        self._coupling_s_outer_kernel = cl.Kernel(self._program, "coupling_s_outer_grad_f32")
        self._coupling_s_outer_clipped_kernel = cl.Kernel(self._program, "coupling_s_outer_grad_clipped_f32")
        self._coupling_input_grad_kernel = cl.Kernel(self._program, "coupling_input_grad_f32")
        self._leaky_relu_kernel = cl.Kernel(self._program, "leaky_relu_f32")
        self._leaky_relu_deriv_kernel = cl.Kernel(self._program, "leaky_relu_deriv_f32")
        self._tanh_deriv_kernel = cl.Kernel(self._program, "tanh_deriv_f32")
        self._fused_bias_act_kernel = cl.Kernel(self._program, "fused_bias_act_f32")
        self._fused_sgd_kernel = cl.Kernel(self._program, "fused_sgd_momentum_f32")
        self._fused_sgd_bias_kernel = cl.Kernel(self._program, "fused_sgd_momentum_bias_f32")
        self._fused_matmul_at_b_clip_kernel = cl.Kernel(self._program, "matmul_at_b_clip_f32")
        self._fused_sum_axis0_clip_kernel = cl.Kernel(self._program, "sum_axis0_clip_f32")
        self._permute_split_kernel = cl.Kernel(self._program, "permute_split_f32")
        self._add3_clip_kernel = cl.Kernel(self._program, "add3_clip_f32")
        self._fused_delta_act_clip_kernel = cl.Kernel(self._program, "fused_delta_act_clip_f32")
        self._concat_invperm_kernel = cl.Kernel(self._program, "concat_invperm_f32")
        self._fused_bias_act_dual_kernel = cl.Kernel(self._program, "fused_bias_act_dual_f32")
        self._fused_forward_invperm_kernel = cl.Kernel(self._program, "fused_forward_invperm_f32")
        self._transpose_kernel = cl.Kernel(self._program, "transpose_f32")
        self._has_pyclblast = pyclblast is not None
        # If CLBlast fails once on this runtime/device, disable and fallback to
        # custom kernel to avoid repeated exception overhead.
        self._disable_pyclblast = False
        self._force_kernel_matmul = False
        
        self.half = vector_size // 2
        self.full = vector_size
        self.batch = batch_size
        self.hidden = internal_width

        # Initializing buffers to reused space
        # self.buffer_batch_full = cl_array.empty(self.queue, (self.batch, self.full), dtype=np.float32)
        # self.buffer_batch_half = cl_array.empty(self.queue, (self.batch, self.half), dtype=np.float32)
        # self.buffer_single_full = cl_array.empty(self.queue, (1, self.full), dtype=np.float32)
        # self.buffer_single_half = cl_array.empty(self.queue, (1, self.half), dtype=np.float32)
        # self.buffer_batch_hidden = cl_array.empty(self.queue, (self.batch, self.hidden), dtype=np.float32)
        # self.buffer_single_hidden = cl_array.empty(self.queue, (1, self.hidden), dtype=np.float32)

        _RING_N = 8  # max simultaneously live buffers of any one shape

        # Replace self.buffer_batch_full / buffer_batch_half / etc. with:
        self._ring_batch_full   = _DeviceBufferRing(self.queue, (self.batch, self.full),   _RING_N)
        self._ring_batch_half   = _DeviceBufferRing(self.queue, (self.batch, self.half),   _RING_N)
        self._ring_single_full  = _DeviceBufferRing(self.queue, (1,          self.full),   _RING_N)
        self._ring_single_half  = _DeviceBufferRing(self.queue, (1,          self.half),   _RING_N)
        self._ring_batch_hidden = _DeviceBufferRing(self.queue, (self.batch, self.hidden), _RING_N)
        self._ring_single_hidden= _DeviceBufferRing(self.queue, (1,          self.hidden), _RING_N)

        # These don't need buffer rings since they are used infrequently.
        self.matmulatb_hidden_half = cl_array.empty(self.queue, (self.hidden, self.half), dtype=np.float32)
        self.matmulatb_half_hidden = cl_array.empty(self.queue, (self.half, self.hidden), dtype=np.float32)



    def set_matmul_engine(self, engine="auto"):
        # auto    -> prefer CLBlast then fallback kernel
        # clblast -> force CLBlast usage (raises if unavailable)
        # kernel  -> force custom OpenCL kernel
        mode = str(engine).lower()
        if mode not in ("auto", "clblast", "kernel"):
            raise ValueError("matmul engine must be one of: auto, clblast, kernel")

        self._force_kernel_matmul = mode == "kernel"
        if mode == "clblast":
            if not self._has_pyclblast:
                raise RuntimeError("pyclblast is not available")
            self._disable_pyclblast = False

    def is_device_array(self, x):
        return cl_array is not None and isinstance(x, cl_array.Array)

    def to_device(self, x, dtype=np.float32):
        # Host arrays are copied as contiguous buffers for predictable kernel access.
        if self.is_device_array(x):
            return x
        host = np.ascontiguousarray(np.asarray(x, dtype=dtype))
        return cl_array.to_device(self.queue, host)

    def to_host(self, x):
        if self.is_device_array(x):
            return x.get()
        return np.asarray(x)

    def matmul(self, a, b):
        # Inputs can be host ndarray or device array; normalize to device float32.
        ad = self.to_device(a, dtype=np.float32)
        bd = self.to_device(b, dtype=np.float32)

        if len(ad.shape) != 2 or len(bd.shape) != 2:
            raise ValueError(f"OpenCL matmul expects 2D arrays, got {ad.shape} and {bd.shape}")

        m, k_a = ad.shape
        k_b, n = bd.shape
        if k_a != k_b:
            raise ValueError(f"OpenCL matmul shape mismatch: {ad.shape} x {bd.shape}")

        if m == 1:
            if n == self.half:   out = self._ring_single_half.next()
            elif n == self.hidden: out = self._ring_single_hidden.next()
            else: out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        elif m == self.batch:
            if n == self.half:   out = self._ring_batch_half.next()
            elif n == self.hidden: out = self._ring_batch_hidden.next()
            else: out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        else:
            out = cl_array.empty(self.queue, (m, n), dtype=np.float32)


        # Fast-path: CLBlast GEMM (typically much faster than the naive kernel).
        # pyclblast expects pyopencl array objects (not raw buffers).
        # Leading dimensions for row-major NumPy-compatible layout:
        # A[M, K] -> a_ld=K, B[K, N] -> b_ld=N, C[M, N] -> c_ld=N.
        if not self._force_kernel_matmul and self._has_pyclblast and not self._disable_pyclblast:
            try:
                pyclblast.gemm(
                    self.queue,
                    int(m),
                    int(n),
                    int(k_a),
                    ad,
                    bd,
                    out,
                    int(k_a),
                    int(n),
                    int(n),
                )
                return out
            except Exception:
                # If CLBlast fails on this runtime/device combination,
                # fallback to custom kernel and avoid repeated failures.
                self._disable_pyclblast = True

        # Fallback kernel: one work-item per output element.
        # global_size = (M, N): OpenCL NDRange over output matrix indices.
        self._matmul_kernel(
            self.queue,
            (int(m), int(n)),
            None,
            ad.data,
            bd.data,
            out.data,
            np.int32(m),
            np.int32(n),
            np.int32(k_a),
        )
        return out

    def add(self, a, b): # Unused
        # Device fast-path when either operand is already on device.
        if self.is_device_array(a) or self.is_device_array(b):
            ad = self.to_device(a)
            bd = self.to_device(b)

            if ad.shape == bd.shape:
                return ad + bd

            # Support (M, N) + (1, N) or (1, N) + (M, N) without host fallback.
            if len(ad.shape) == 2 and len(bd.shape) == 2:
                if ad.shape[1] == bd.shape[1]:
                    if bd.shape[0] == 1 and ad.shape[0] >= 1:
                        m, n = ad.shape
                        out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
                        print(m,n)
                        self._add_row_broadcast_kernel(
                            self.queue,
                            (int(m), int(n)),
                            None,
                            ad.data,
                            bd.data,
                            out.data,
                            np.int32(m),
                            np.int32(n),
                        )
                        return out
                    if ad.shape[0] == 1 and bd.shape[0] >= 1:
                        m, n = bd.shape
                        print(m,n)
                        out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
                        self._add_row_broadcast_kernel(
                            self.queue,
                            (int(m), int(n)),
                            None,
                            bd.data,
                            ad.data,
                            out.data,
                            np.int32(m),
                            np.int32(n),
                        )
                        return out

            # Fallback for unsupported broadcast patterns (done on host, re-uploaded).
            return self.to_device(np.add(self.to_host(ad), self.to_host(bd)), dtype=np.float32)
        return np.add(a, b)

    def subtract(self, a, b): # Unused
        if np.isscalar(a) and self.is_device_array(b):
            return float(a) - b
        if self.is_device_array(a) and np.isscalar(b):
            return a - float(b)
        if self.is_device_array(a) or self.is_device_array(b):
            ad = self.to_device(a)
            bd = self.to_device(b)
            return ad - bd
        return np.subtract(a, b)

    def multiply(self, a, b):
        if np.isscalar(a) and self.is_device_array(b):
            return float(a) * b
        if self.is_device_array(a) and np.isscalar(b):
            return a * float(b)
        if self.is_device_array(a) or self.is_device_array(b):
            ad = self.to_device(a)
            bd = self.to_device(b)
            return ad * bd
        return np.multiply(a, b)

    def divide(self, a, b):
        if np.isscalar(a) and self.is_device_array(b):
            return float(a) / b
        if self.is_device_array(a) and np.isscalar(b):
            return a / float(b)
        if self.is_device_array(a) or self.is_device_array(b):
            ad = self.to_device(a)
            bd = self.to_device(b)
            return ad / bd
        return np.divide(a, b)
    
    def soft_clip(self, x, limit=2.0):
        """Fused soft_clip: limit * tanh(x / limit) in a single kernel."""
        xd = self.to_device(x, dtype=np.float32)
        out = cl_array.empty(self.queue, xd.shape, dtype=np.float32)
        n = xd.size
        self._soft_clip_kernel(
            self.queue, (int(n),), None,
            xd.data, out.data, np.float32(limit), np.int32(n),
        )
        return out
    
    def sum(self, x, axis=None, keepdims=False): # Device path unused
        if self.is_device_array(x):
            if axis is None:
                result = cl_array.sum(x)
                if keepdims:
                    # result is a scalar device array with shape (1,); reshape to match input ndim
                    return result.reshape((1,) * x.ndim)
                return result
            if axis == 0 and len(x.shape) == 2:
                m, n = x.shape


                out = cl_array.empty(self.queue, (1, n) if keepdims else (n,), dtype=np.float32)
                self._sum_axis0_kernel(
                    self.queue, (int(n),), None,
                    x.data, out.data, np.int32(m), np.int32(n),
                )
                return out
            result = np.sum(x.get(), axis=axis, keepdims=keepdims)
            return self.to_device(result, dtype=np.float32)
        return np.sum(x, axis=axis, keepdims=keepdims)

    def coupling_forward(self, u, s_raw, t, limit=2.0):
        """v = u * exp(soft_clip(s_raw, limit)) + t — single kernel."""
        ud = self.to_device(u, dtype=np.float32)
        sd = self.to_device(s_raw, dtype=np.float32)
        td = self.to_device(t, dtype=np.float32)


        m, n = ud.shape
        if (m == self.batch and n==self.half):
            out = self._ring_batch_half.next()
            # out = self.buffer_batch_half
        elif (m == 1 and n==self.half):
            out = self._ring_single_half.next()
            # out = self.buffer_single_half
        else:
            out = cl_array.empty(self.queue, ud.shape, dtype=np.float32)
        n = ud.size
        # out = cl_array.empty(self.queue, ud.shape, dtype=np.float32)
        self._coupling_forward_kernel(
            self.queue, (int(n),), None,
            ud.data, sd.data, td.data, out.data,
            np.float32(limit), np.int32(n),
        )
        return out

    def coupling_backward(self, v, s_raw, t, limit=2.0):
        """u = (v - t) * exp(-soft_clip(s_raw, limit)) — single kernel."""
        vd = self.to_device(v, dtype=np.float32)
        sd = self.to_device(s_raw, dtype=np.float32)
        td = self.to_device(t, dtype=np.float32)
        n = vd.size
        out = cl_array.empty(self.queue, vd.shape, dtype=np.float32)
        self._coupling_backward_kernel(
            self.queue, (int(n),), None,
            vd.data, sd.data, td.data, out.data,
            np.float32(limit), np.int32(n),
        )
        return out

    def coupling_s_outer_grad(self, diff, u, s_raw, s_limit=2.0, clip_limit=0.0):
        """outer = diff * u * exp(soft_clip(s, s_limit)) * soft_clip_deriv(s, s_limit)
        Optionally applies soft_clip(result, clip_limit) if clip_limit > 0."""
        dd = self.to_device(diff, dtype=np.float32)
        ud = self.to_device(u, dtype=np.float32)
        sd = self.to_device(s_raw, dtype=np.float32)
        m, n = dd.shape
        if (m == self.batch and n==self.half):
            out = self._ring_batch_half.next()
            # out = self.buffer_batch_half
        elif (m == 1 and n==self.half):
            out = self._ring_single_half.next()
            # out = self.buffer_single_half
        else:
            out = cl_array.empty(self.queue, dd.shape, dtype=np.float32)
        n = dd.size
        # out = cl_array.empty(self.queue, dd.shape, dtype=np.float32)
        if clip_limit > 0:
            self._coupling_s_outer_clipped_kernel(
                self.queue, (int(n),), None,
                dd.data, ud.data, sd.data, out.data,
                np.float32(s_limit), np.float32(clip_limit), np.int32(n),
            )
        else:
            self._coupling_s_outer_kernel(
                self.queue, (int(n),), None,
                dd.data, ud.data, sd.data, out.data,
                np.float32(s_limit), np.int32(n),
            )
        return out

    def coupling_input_grad(self, diff, s_raw, limit=2.0):
        """du = diff * exp(soft_clip(s_raw, limit)) — single kernel."""
        dd = self.to_device(diff, dtype=np.float32)
        sd = self.to_device(s_raw, dtype=np.float32)
        m, n = dd.shape
        if (m == self.batch and n==self.half):
            out = self._ring_batch_half.next()
            # out = self.buffer_batch_half
        else:
            out = cl_array.empty(self.queue, dd.shape, dtype=np.float32)
        n = dd.size
        self._coupling_input_grad_kernel(
            self.queue, (int(n),), None,
            dd.data, sd.data, out.data,
            np.float32(limit), np.int32(n),
        )
        return out

    def apply_activation(self, x, func_name): # Unused?
        xd = self.to_device(x)
        if func_name == "leakyReLU":
            n = xd.size
            out = cl_array.empty(self.queue, xd.shape, dtype=np.float32)
            self._leaky_relu_kernel(
                self.queue, (int(n),), None,
                xd.data, out.data, np.float32(0.01), np.int32(n),
            )
            return out
        if func_name == "tanh":
            return clmath.tanh(xd)
        if func_name == "identity":
            return xd
        return None

    def apply_activation_derivative(self, x, func_name): # Unused?
        xd = self.to_device(x)
        if func_name == "leakyReLU":
            n = xd.size
            out = cl_array.empty(self.queue, xd.shape, dtype=np.float32)
            self._leaky_relu_deriv_kernel(
                self.queue, (int(n),), None,
                xd.data, out.data, np.float32(0.01), np.int32(n),
            )
            return out
        if func_name == "tanh":
            n = xd.size
            out = cl_array.empty(self.queue, xd.shape, dtype=np.float32)
            self._tanh_deriv_kernel(
                self.queue, (int(n),), None,
                xd.data, out.data, np.int32(n),
            )
            return out
        if func_name == "identity":
            out = cl_array.empty(self.queue, xd.shape, dtype=np.float32)
            out.fill(np.float32(1.0))
            return out
        return None

    def transpose(self, x):
        """Return a contiguous transposed copy on device."""
        if self.is_device_array(x):
            m, n = x.shape
            out = cl_array.empty(self.queue, (n, m), dtype=np.float32)
            self._transpose_kernel(
                self.queue, (int(m), int(n)), None,
                x.data, out.data,
                np.int32(m), np.int32(n),
            )
            return out
        return np.ascontiguousarray(np.asarray(x).T)

    def fused_bias_act(self, x, bias, act_type):
        """Fused bias-add + activation: activation(X + B) in one kernel."""
        xd = self.to_device(x, dtype=np.float32)
        bd = self.to_device(bias, dtype=np.float32)
        m, n = xd.shape
        out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        if act_type < 0:
            return None  # unsupported — caller falls back
        self._fused_bias_act_kernel(
            self.queue, (int(m), int(n)), None,
            xd.data, bd.data, out.data,
            np.int32(m), np.int32(n), np.int32(act_type),
        )
        return out
    
    def fused_sgd_update(self, weights, velocity, gradient, lr, momentum, decay_factor):
        """In`-place: v = mom*v + g; w = decay*w + lr*v"""
        n = weights.size
        self._fused_sgd_kernel(
            self.queue, (int(n),), None,
            weights.data, velocity.data, gradient.data,
            np.float32(lr), np.float32(momentum), np.float32(decay_factor),
            np.int32(n),
        )

    def fused_sgd_bias_update(self, bias, velocity, gradient, lr, momentum):
        """In-place: v = mom*v + g; b = b + lr*v"""
        n = bias.size
        self._fused_sgd_bias_kernel(
            self.queue, (int(n),), None,
            bias.data, velocity.data, gradient.data,
            np.float32(lr), np.float32(momentum),
            np.int32(n),
        )

    def matmul_at_b_clip(self, a, b, clip):
        """Compute clip(A^T @ B) without materializing the transpose."""
        ad = self.to_device(a, dtype=np.float32)
        bd = self.to_device(b, dtype=np.float32)
        # A is (K, M), A^T is (M, K), B is (K, N), result is (M, N)
        k, m = ad.shape
        k2, n = bd.shape
        if k != k2:
            raise ValueError(f"matmul_at_b shape mismatch: A ({k},{m}) B ({k2},{n})")
        # print(m, n)
        if (m == self.hidden and n == self.half):
            out = self.matmulatb_hidden_half
        elif (m == self.half and n == self.hidden):
            out = self.matmulatb_half_hidden
        else:
            out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        self._fused_matmul_at_b_clip_kernel(
            self.queue, (int(m), int(n)), None,
            ad.data, bd.data, out.data,
            np.int32(m), np.int32(n), np.int32(k), np.float32(clip),
        )
        return out

    def sum_axis0_clip(self, x, clip):
        """Compute sum along axis 0 with clipping."""
        m, n = x.shape
        if (n==self.half):
            out = self._ring_single_half.next()
            # out = self.buffer_batch_half
        elif (n==self.hidden):
            out = self._ring_single_hidden.next()
            # out = self.buffer_single_half
        else:
            out = cl_array.empty(self.queue, (1, n), dtype=np.float32)
        self._fused_sum_axis0_clip_kernel(
            self.queue, (int(n),), None,
            x.data, out.data,
            np.int32(m), np.int32(n), np.float32(clip),
        )
        return out
    
    def delta_act_clip(self, inputs, outputs, act_type, clip):
        """Compute input_delta * act_deriv(preact) with optional clipping in one kernel.
        inputs: pre-activation values (for computing act_deriv)"""
        m, n = inputs.shape
        # print(m, n)
        if (m == self.batch and n==self.half):
            out = self._ring_batch_half.next()
            # out = self.buffer_batch_half
        elif (m == self.batch and n==self.hidden):
            out = self._ring_batch_hidden.next()
            # out = self.buffer_batch_hidden
        else:
            out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        # out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        self._fused_delta_act_clip_kernel(
            self.queue, (int(n*m),), None,
            inputs.data, outputs.data, out.data, np.int32(act_type), np.float32(clip), np.int32(n*m),
        )
        return out
    
    def add3_clip(self, A, B, C, clip):
        """Compute sum along axis 0 with clipping."""
        m, n = A.shape
        # print(m, n)
        if (m == self.batch and n==self.half):
            out = self._ring_batch_half.next()
            # out = self.buffer_batch_half
        else:
            out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        # out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        self._add3_clip_kernel(
            self.queue, (int(n*m),), None,
            A.data, B.data, C.data, out.data, np.float32(clip), np.int32(n*m),
        )
        return out

    def permute_split(self, x, perm):
        """Permutes and splits the input array."""
        x = self.to_device(x, dtype=np.float32)
        m, n = x.shape
        if (m == self.batch and n==self.half):
            out1 = self._ring_batch_half.next()
            out2 = self._ring_batch_half.next()
        elif (m == 1 and n==self.half):
            out1 = self._ring_single_half.next()
            out2 = self._ring_single_half.next()
        else:
            out1 = cl_array.empty(self.queue, (m, n//2), dtype=np.float32)
            out2 = cl_array.empty(self.queue, (m, n//2), dtype=np.float32)
        self._permute_split_kernel(
            self.queue, (int(m), int(n)), None,
            x.data, perm.data, out1.data, out2.data, np.int32(m), np.int32(n), np.int32(n//2),
        )
        return out1, out2
    
    def concat_invperm(self, x1, x2, inv_perm):
        """Concatenates and applies inverse permutation to the input arrays."""
        x1 = self.to_device(x1, dtype=np.float32)
        x2 = self.to_device(x2, dtype=np.float32)
        m, n = x1.shape
        perm_dev = self.to_device(inv_perm, dtype=np.int32)
        if (m == self.batch and n==self.full):
            out = self._ring_batch_full.next()
        elif (m == 1 and n==self.full):
            out = self._ring_batch_full.next()
        else:
            out = cl_array.empty(self.queue, (m, 2*n), dtype=np.float32)
        self._concat_invperm_kernel(
            self.queue, (int(m), int(2*n)), None,
            x1.data, x2.data, perm_dev.data, out.data, np.int32(m), np.int32(n),
        )
        return out

    def bias_act_dual(self, x, bias, act_type):
        """Applies bias and activation function while saving the results before and after activation."""
        x = self.to_device(x, dtype=np.float32)
        bias = self.to_device(bias, dtype=np.float32)
        m, n = x.shape
        out1 = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        out2 = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        # print(f"Launching fused_bias_act_dual kernel with M={m}, N={n}, act_type={act_type}, x_ptr={x.data}, bias_ptr={bias.data}, out1_ptr={out1.data}, out2_ptr={out2.data}")
        self._fused_bias_act_dual_kernel(
            self.queue, (int(m), int(n)), None,
            x.data, bias.data, out1.data, out2.data, np.int32(m), np.int32(n), np.int32(act_type),
        )
        return out1, out2
    
    def fused_forward_invperm(self, v1, u2, inv_perm, s_raw, t, limit=2.0):
        """Concatenates and applies inverse permutation to the input arrays in a fused kernel."""
        v1 = self.to_device(v1, dtype=np.float32)
        u2 = self.to_device(u2, dtype=np.float32)
        s_raw = self.to_device(s_raw, dtype=np.float32)
        t = self.to_device(t, dtype=np.float32)

        m, n = v1.shape
        perm_dev = self.to_device(inv_perm, dtype=np.int32)
        # print(m, 2*n)
        if (m == self.batch and n==self.full):
            out = self._ring_batch_full.next()
            # out = self.buffer_batch_half
        elif (m == 1 and n==self.full):
            out = self._ring_batch_full.next()
            # out = self.buffer_batch_half
        else:
            out = cl_array.empty(self.queue, (m, 2*n), dtype=np.float32)
        # out = cl_array.empty(self.queue, (m, 2*n), dtype=np.float32)
        self._fused_forward_invperm_kernel(
            self.queue, (int(m), int(2*n)), None,
            v1.data, u2.data, perm_dev.data, s_raw.data, t.data, out.data, np.int32(m), np.int32(n), np.float32(limit),
        )
        return out