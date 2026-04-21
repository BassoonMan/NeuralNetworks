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

    // Backward coupling: u = (v - t) * exp(-soft_clip(s_raw, limit))
    __kernel void coupling_backward_f32(
        __global const float* V,
        __global const float* ST_raw,
        __global float* U,
        const float limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        float s_clipped = limit * tanh(ST_raw[i] / limit);
        U[i] = (V[i] - ST_raw[i + N]) * exp(-s_clipped);
    }

    // Input gradient: du = diff * exp(soft_clip(s_raw, limit))
    __kernel void coupling_input_grad_f32(
        __global const float* diff,
        __global const float* ST_raw,
        __global float* du,
        const float limit,
        const int N,
        const int cols)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        const int row = i / (cols / 2);
        float s_clipped = limit * tanh(ST_raw[row * cols + (i % (cols / 2))] / limit);
        du[i] = diff[i] * exp(s_clipped);
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
        } else if (act_type == 3) {
            val = (col < N / 2) ? tanh(val) : val;
        }
        // act_type == 0: identity (no-op)

        Y[row * N + col] = val;
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
    // Tiled GEMM — 16×16 tiles with local memory for data reuse.
    // K = BATCH (compile-time #define).  Padding +1 avoids bank conflicts.
    #define ATB_TILE 16
    __kernel void matmul_at_b_clip_f32(
        __global const float* A,
        __global const float* B,
        __global float* C,
        const int M,
        const int N,
        const float clip_limit)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        const int lr  = get_local_id(0);
        const int lc  = get_local_id(1);

        __local float tileA[ATB_TILE][ATB_TILE + 1];
        __local float tileB[ATB_TILE][ATB_TILE + 1];

        float acc = 0.0f;

        for (int t = 0; t < BATCH; t += ATB_TILE) {
            // tileA[lr][lc] = A^T[row, t+lc] = A[(t+lc)*M + row]
            int ka = t + lc;
            tileA[lr][lc] = (row < M && ka < BATCH) ? A[ka * M + row] : 0.0f;

            // tileB[lr][lc] = B[t+lr, col]
            int kb = t + lr;
            tileB[lr][lc] = (kb < BATCH && col < N) ? B[kb * N + col] : 0.0f;

            barrier(CLK_LOCAL_MEM_FENCE);

            #pragma unroll
            for (int kk = 0; kk < ATB_TILE; ++kk) {
                acc += tileA[lr][kk] * tileB[kk][lc];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (row < M && col < N) {
            if (clip_limit > 0.0f) {
                acc = clip_limit * tanh(acc / clip_limit);
            }
            C[row * N + col] = acc;
        }
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
        const int N,
        const int cols)
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
        } else if (act_type == 3) {
            int col = i % cols;
            d = (col < cols / 2) ? (1.0f - tanh(preact[i]) * tanh(preact[i])) : 1.0f;
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
        __global const float* ST_raw,
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
            s_clipped = limit * tanh(ST_raw[row * 2 * half_N + (src_col - half_N)] / limit);
            val = U2[row * half_N + (src_col - half_N)] * exp(s_clipped) + ST_raw[row * 2 * half_N + src_col];
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
        } else if (act_type == 3) {
            val = (col < N / 2) ? tanh(val) : val;
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

    __kernel void coupling_forward_merged_f32(
        __global const float* U,       // (M, half)
        __global const float* ST_raw,  // (M, 2*half) — first half = s, second half = t
        __global float* V,             // (M, half)
        const float limit,
        const int M,
        const int half_N)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= half_N) return;
        
        float s_val = ST_raw[row * (2 * half_N) + col];
        float t_val = ST_raw[row * (2 * half_N) + half_N + col];
        float s_clipped = limit * tanh(s_val / limit);
        V[row * half_N + col] = U[row * half_N + col] * exp(s_clipped) + t_val;
    }


        // Fused: out[i] = (A[i]-B[i])/D, partial_sums[wg] = sum_within_workgroup(out[i]^2)
    __kernel void subtract_divide_loss_f32(
        __global const float* A,
        __global const float* B,
        __global float* out,
        __global float* partial_sums,
        const int total,
        const float D)
    {
        const int gid = get_global_id(0);
        const int lid = get_local_id(0);
        const int group_size = get_local_size(0);

        __local float scratch[256];

        float sq = 0.0f;
        if (gid < total) {
            float diff = (A[gid] - B[gid]) / D;
            out[gid] = diff;
            sq = diff * diff;
        }

        scratch[lid] = sq;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int stride = group_size >> 1; stride > 0; stride >>= 1) {
            if (lid < stride) {
                scratch[lid] += scratch[lid + stride];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (lid == 0) {
            partial_sums[get_group_id(0)] = scratch[0];
        }
    }

    __kernel void fused_sgd_with_transpose_f32(
        __global float* W,          // (M, N) weights — read/write
        __global float* W_T,        // (N, M) transposed weights — write
        __global float* V,          // (M*N) velocity — read/write
        __global const float* G,    // (M*N) gradient — read
        const float lr,
        const float momentum,
        const float decay_factor,
        const int M,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= M * N) return;

        // Standard SGD+momentum
        float v = momentum * V[i] + G[i];
        V[i] = v;
        float w_new = decay_factor * W[i] + lr * v;
        W[i] = w_new;

        // Write into transpose position simultaneously
        int row = i / N;
        int col = i % N;
        W_T[col * M + row] = w_new;
    }

    // Fused: build both diffs_st1 and diffs_st2 in one pass.
    //   diffs_st1[:, :half] = s1_outer                          (copy)
    //   diffs_st1[:, half:] = soft_clip(diff2, t_clip)          (elementwise)
    //   diffs_st2[:, :half] = s_outer_grad(diff1_total, u1, st2_raw)  (coupling grad)
    //   diffs_st2[:, half:] = soft_clip(diff1_total, t_clip)    (elementwise)
    // Replaces: coupling_s_outer_grad_merged + 2× soft_clip + 2× concat_cols
    __kernel void fused_coupling_grads_concat_f32(
        __global const float* s1_outer,      // (M, half_N)
        __global const float* diff2,         // (M, half_N)
        __global const float* diff1_total,   // (M, half_N)
        __global const float* u1,            // (M, half_N)
        __global const float* st2_raw,       // (M, 2*half_N)
        __global float* diffs_st1,           // (M, 2*half_N) output
        __global float* diffs_st2,           // (M, 2*half_N) output
        const float s_limit,
        const float s_clip_limit,
        const float t_clip_limit,
        const int M,
        const int half_N)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        if (row >= M || col >= 2 * half_N) return;

        const int out_idx = row * (2 * half_N) + col;

        if (col < half_N) {
            const int src_idx = row * half_N + col;

            // diffs_st1 left half: just copy s1_outer
            diffs_st1[out_idx] = s1_outer[src_idx];

            // diffs_st2 left half: coupling_s_outer_grad from st2_raw
            float d = diff1_total[src_idx];
            float u = u1[src_idx];
            float s_raw_val = st2_raw[row * (2 * half_N) + col];
            float scaled = s_raw_val / s_limit;
            float th = tanh(scaled);
            float s_clipped = s_limit * th;
            float clip_deriv = 1.0f - th * th;
            float val = d * u * exp(s_clipped) * clip_deriv;
            if (s_clip_limit > 0.0f) {
                val = s_clip_limit * tanh(val / s_clip_limit);
            }
            diffs_st2[out_idx] = val;
        } else {
            const int src_idx = row * half_N + (col - half_N);

            // diffs_st1 right half: soft_clip(diff2, t_clip)
            float d2 = diff2[src_idx];
            diffs_st1[out_idx] = (t_clip_limit > 0.0f)
                ? t_clip_limit * tanh(d2 / t_clip_limit)
                : d2;

            // diffs_st2 right half: soft_clip(diff1_total, t_clip)
            float d1 = diff1_total[src_idx];
            diffs_st2[out_idx] = (t_clip_limit > 0.0f)
                ? t_clip_limit * tanh(d1 / t_clip_limit)
                : d1;
        }
    }

    // Fused coupling_s_outer_grad_merged + concat_cols.
    // Left half of combined output = s_outer_grad(diff, U, ST_raw).
    // Right half of combined output = copy of diff.
    // Also writes s_outer to a separate buffer (needed later by fused_coupling_grads_concat).
    // Global size: (M, 2 * half_N) — same as concat_cols.
    __kernel void coupling_s_outer_concat_f32(
        __global const float* diff,          // (M, half_N)
        __global const float* U,             // (M, half_N)
        __global const float* ST_raw,        // (M, 2*half_N)
        __global float* out_combined,        // (M, 2*half_N) — [s_outer | diff]
        __global float* out_s_outer,         // (M, half_N)   — s_outer only
        const float s_limit,
        const float clip_limit,
        const int M,
        const int half_N)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        const int full_N = 2 * half_N;
        if (row >= M || col >= full_N) return;

        const int out_idx = row * full_N + col;

        if (col < half_N) {
            // Left half: compute s_outer_grad
            const int half_idx = row * half_N + col;
            float s_raw_val = ST_raw[row * full_N + col];
            float scaled = s_raw_val / s_limit;
            float th = tanh(scaled);
            float s_clipped = s_limit * th;
            float clip_deriv = 1.0f - th * th;
            float val = diff[half_idx] * U[half_idx] * exp(s_clipped) * clip_deriv;

            if (clip_limit > 0.0f) {
                val = clip_limit * tanh(val / clip_limit);
            }

            out_combined[out_idx] = val;
            out_s_outer[half_idx] = val;
        } else {
            // Right half: copy diff
            out_combined[out_idx] = diff[row * half_N + (col - half_N)];
        }
    }

    // out = soft_clip(A + B, limit)
    __kernel void add2_clip_f32(
        __global const float* A,
        __global const float* B,
        __global float* out,
        const float clip_limit,
        const int N)
    {
        const int i = get_global_id(0);
        if (i >= N) return;

        float val = A[i] + B[i];

        if (clip_limit > 0.0f) {
            val = clip_limit * tanh(val / clip_limit);
        }

        out[i] = val;
    }

    // Backward coupling outer gradient: u = (v - t) * exp(-s_clip)
    // Left half  (col < half): soft_clip(-diff * u * s_clip_deriv, s_clip_limit)
    // Right half (col >= half): soft_clip(-diff * exp(-s_clip), t_clip_limit)
    __kernel void coupling_backward_outer_concat_f32(
        __global const float* diff,      // (M, half_N)
        __global const float* U,         // (M, half_N)
        __global const float* ST_raw,    // (M, 2*half_N)
        __global float* out,             // (M, 2*half_N)
        const float s_limit,
        const float s_clip_limit,
        const float t_clip_limit,
        const int M,
        const int half_N)
    {
        const int row = get_global_id(0);
        const int col = get_global_id(1);
        const int full_N = 2 * half_N;
        if (row >= M || col >= full_N) return;

        const int out_idx = row * full_N + col;

        if (col < half_N) {
            const int half_idx = row * half_N + col;
            float s_raw_val = ST_raw[row * full_N + col];
            float th = tanh(s_raw_val / s_limit);
            float clip_deriv = 1.0f - th * th;
            float val = -diff[half_idx] * U[half_idx] * clip_deriv;
            if (s_clip_limit > 0.0f) {
                val = s_clip_limit * tanh(val / s_clip_limit);
            }
            out[out_idx] = val;
        } else {
            const int half_idx = row * half_N + (col - half_N);
            float s_raw_val = ST_raw[row * full_N + (col - half_N)];
            float s_clipped = s_limit * tanh(s_raw_val / s_limit);
            float val = -diff[half_idx] * exp(-s_clipped);
            if (t_clip_limit > 0.0f) {
                val = t_clip_limit * tanh(val / t_clip_limit);
            }
            out[out_idx] = val;
        }
    }

    // Backward coupling input gradient: dv = diff * exp(-s_clip)
    __kernel void coupling_backward_input_grad_merged_f32(
        __global const float* diff,
        __global const float* ST_raw,
        __global float* du,
        const float limit,
        const int N,
        const int cols)
    {
        const int i = get_global_id(0);
        if (i >= N) return;
        const int row = i / (cols / 2);
        float s_clipped = limit * tanh(ST_raw[row * cols + (i % (cols / 2))] / limit);
        du[i] = diff[i] * exp(-s_clipped);
    }

    """

    def __init__(self, batch_size=128, vector_size=64, internal_width=64, preferred_device_type="gpu", coupling_layers=10, internal_network_layers=2):
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
        self.half = vector_size // 2
        self.full = vector_size
        self.batch = batch_size
        self.hidden = internal_width

        # Tiled matmul_at_b_clip sizes (round up to multiple of 16)
        _ATB_TILE = 16
        self._atb_local = (_ATB_TILE, _ATB_TILE)
        self._gs_half_hid2_tiled = (
            int(((self.half + _ATB_TILE - 1) // _ATB_TILE) * _ATB_TILE),
            int(((self.hidden * 2 + _ATB_TILE - 1) // _ATB_TILE) * _ATB_TILE),
        )
        self._gs_hid2_full_tiled = (
            int(((self.hidden * 2 + _ATB_TILE - 1) // _ATB_TILE) * _ATB_TILE),
            int(((self.full + _ATB_TILE - 1) // _ATB_TILE) * _ATB_TILE),
        )


        self._dim_header = f"""
            #define BATCH {self.batch}
            #define FULL_N {self.full}
            #define HALF_N {self.half}
            #define HIDDEN {self.hidden}
            #define HIDDENX2 {self.hidden * 2}
            """

        self.context = cl.Context([selected_device])
        self.queue = cl.CommandQueue(self.context)
        self._program = cl.Program(self.context, self._dim_header + self._GEMM_SOURCE + self._FUSED_SOURCE).build()
        self._matmul_kernel = cl.Kernel(self._program, "matmul_f32")
        self._add_row_broadcast_kernel = cl.Kernel(self._program, "add_row_broadcast_f32")
        self._soft_clip_kernel = cl.Kernel(self._program, "soft_clip_f32")
        self._coupling_backward_kernel = cl.Kernel(self._program, "coupling_backward_f32")
        self._coupling_input_grad_kernel = cl.Kernel(self._program, "coupling_input_grad_f32")
        self._leaky_relu_kernel = cl.Kernel(self._program, "leaky_relu_f32")
        self._leaky_relu_deriv_kernel = cl.Kernel(self._program, "leaky_relu_deriv_f32")
        self._tanh_deriv_kernel = cl.Kernel(self._program, "tanh_deriv_f32")
        self._fused_bias_act_kernel = cl.Kernel(self._program, "fused_bias_act_f32")
        self._fused_sgd_bias_kernel = cl.Kernel(self._program, "fused_sgd_momentum_bias_f32")
        self._fused_matmul_at_b_clip_kernel = cl.Kernel(self._program, "matmul_at_b_clip_f32")
        self._fused_sum_axis0_clip_kernel = cl.Kernel(self._program, "sum_axis0_clip_f32")
        self._permute_split_kernel = cl.Kernel(self._program, "permute_split_f32")
        self._fused_delta_act_clip_kernel = cl.Kernel(self._program, "fused_delta_act_clip_f32")
        self._concat_invperm_kernel = cl.Kernel(self._program, "concat_invperm_f32")
        self._fused_bias_act_dual_kernel = cl.Kernel(self._program, "fused_bias_act_dual_f32")
        self._fused_forward_invperm_kernel = cl.Kernel(self._program, "fused_forward_invperm_f32")
        self._transpose_kernel = cl.Kernel(self._program, "transpose_f32")
        self._coupling_forward_merged_kernel = cl.Kernel(self._program, "coupling_forward_merged_f32")
        self._subtract_divide_loss_kernel = cl.Kernel(self._program, "subtract_divide_loss_f32")
        self._fused_sgd_transpose_kernel = cl.Kernel(self._program, "fused_sgd_with_transpose_f32")
        self._fused_coupling_grads_concat_kernel = cl.Kernel(self._program, "fused_coupling_grads_concat_f32")
        self._coupling_s_outer_concat_kernel = cl.Kernel(self._program, "coupling_s_outer_concat_f32")
        self._add2_clip_kernel = cl.Kernel(self._program, "add2_clip_f32")
        self._coupling_backward_outer_concat_kernel = cl.Kernel(self._program, "coupling_backward_outer_concat_f32")
        self._coupling_backward_input_grad_merged_kernel = cl.Kernel(self._program, "coupling_backward_input_grad_merged_f32")
        self._has_pyclblast = pyclblast is not None
        # If CLBlast fails once on this runtime/device, disable and fallback to
        # custom kernel to avoid repeated exception overhead.
        self._disable_pyclblast = False
        self._force_kernel_matmul = False
    


        # Initializing buffers to reused space
        _RING_N = 16  # max simultaneously live buffers of any one shape
        n_subnets = coupling_layers * 2
        n_hidden = internal_network_layers - 1

        self._ring_batch_full   = _DeviceBufferRing(self.queue, (self.batch, self.full),   n_subnets * 2 + coupling_layers + 16)
        self._ring_batch_half   = _DeviceBufferRing(self.queue, (self.batch, self.half),   coupling_layers * 3 + 16)
        self._ring_single_full  = _DeviceBufferRing(self.queue, (1,          self.full),   _RING_N)
        self._ring_single_half  = _DeviceBufferRing(self.queue, (1,          self.half),   _RING_N)
        self._ring_batch_hidden = _DeviceBufferRing(self.queue, (self.batch, self.hidden), _RING_N)
        self._ring_batch_hiddenx2 = _DeviceBufferRing(self.queue, (self.batch, self.hidden * 2), n_subnets * (3 * n_hidden))
        self._ring_single_hidden= _DeviceBufferRing(self.queue, (1,          self.hidden), _RING_N)
        self._ring_single_hiddenx2= _DeviceBufferRing(self.queue, (1,          self.hidden * 2), _RING_N)
        self._ring_batch_batch  = _DeviceBufferRing(self.queue, (self.batch, self.batch),  _RING_N*2)

        # These don't need buffer rings since they are used infrequently.
        self.matmulatb_half_hidden = cl_array.empty(self.queue, (self.half, self.hidden * 2), dtype=np.float32)
        self.matmulatb_hidden_full = cl_array.empty(self.queue, (self.hidden * 2, self.full), dtype=np.float32)

        self._prebind_batch_kernels()


    # Currently working on glitch with buffer rings for probably hiddenx2 dimensions.
    def _prebind_batch_kernels(self):
        """Create pre-bound kernel instances for training hot path.
        Fixed scalar args (dimensions, limits) are set once here.
        Only buffer pointers need set_arg() per call."""

        # Cached np scalars — avoids per-call np.int32/float32 construction
        self._c_batch      = np.int32(self.batch)
        self._c_full       = np.int32(self.full)
        self._c_half       = np.int32(self.half)
        self._c_hidden2    = np.int32(self.hidden * 2)
        self._c_limit2     = np.float32(2.0)
        self._c_bxhalf     = np.int32(self.batch * self.half)
        self._c_bxfull     = np.int32(self.batch * self.full)
        self._c_bxhidden2  = np.int32(self.batch * self.hidden * 2)
        self._c_act        = [np.int32(i) for i in range(4)]  # act_type 0-3

        # Cached global_size tuples
        self._gs_b_full      = (int(self.batch), int(self.full))
        self._gs_b_half      = (int(self.batch), int(self.half))
        self._gs_b_hidden2   = (int(self.batch), int(self.hidden * 2))
        self._gs_bxhalf      = (self.batch * self.half,)
        self._gs_bxfull      = (self.batch * self.full,)
        self._gs_bxhidden2   = (self.batch * self.hidden * 2,)
        self._gs_half_hid2   = (int(self.half), int(self.hidden * 2))
        self._gs_hid2_full   = (int(self.hidden * 2), int(self.full))
        self._gs_hidden2     = (int(self.hidden * 2),)
        self._gs_full        = (int(self.full),)

        # Float32 cache for common clip/limit values
        self._cf = {}

        # ---- Coupling-layer kernels ----

        # permute_split_f32(X, perm, U1, U2, M, full_N, half_N)
        k = cl.Kernel(self._program, "permute_split_f32")
        k.set_arg(4, self._c_batch)
        k.set_arg(5, self._c_full)
        k.set_arg(6, self._c_half)
        self._pb_permute_split = k

        # coupling_forward_merged_f32(U, ST, V, limit, M, half_N)
        k = cl.Kernel(self._program, "coupling_forward_merged_f32")
        k.set_arg(3, self._c_limit2)
        k.set_arg(4, self._c_batch)
        k.set_arg(5, self._c_half)
        self._pb_coupling_fwd_merged = k

        # fused_forward_invperm_f32(V1, U2, inv_perm, ST, out, M, half_N, limit)
        k = cl.Kernel(self._program, "fused_forward_invperm_f32")
        k.set_arg(5, self._c_batch)
        k.set_arg(6, self._c_half)
        k.set_arg(7, self._c_limit2)
        self._pb_fwd_invperm = k

        # concat_invperm_f32(U1, U2, inv_perm, out, M, half_N)
        k = cl.Kernel(self._program, "concat_invperm_f32")
        k.set_arg(4, self._c_batch)
        k.set_arg(5, self._c_half)
        self._pb_concat_invperm = k

        # coupling_input_grad_f32(diff, ST, du, limit, N, cols)
        k = cl.Kernel(self._program, "coupling_input_grad_f32")
        k.set_arg(3, self._c_limit2)
        k.set_arg(4, self._c_bxhalf)       # N = batch*half
        k.set_arg(5, self._c_full)          # cols = st_raw width
        self._pb_input_grad = k

        # coupling_backward_f32(V, ST, U, limit, N)
        k = cl.Kernel(self._program, "coupling_backward_f32")
        k.set_arg(3, self._c_limit2)
        k.set_arg(4, self._c_bxhalf)
        self._pb_coupling_bwd = k

        # ---- Subnetwork kernels (two variants per: hidden-layer dims, output-layer dims) ----

        # bias_act_dual_f32(X, B, pre, post, M, N, act_type)
        k = cl.Kernel(self._program, "fused_bias_act_dual_f32")
        k.set_arg(4, self._c_batch);  k.set_arg(5, self._c_hidden2)
        self._pb_bad_hid = k

        k = cl.Kernel(self._program, "fused_bias_act_dual_f32")
        k.set_arg(4, self._c_batch);  k.set_arg(5, self._c_full)
        self._pb_bad_full = k

        # delta_act_clip_f32(inp, pre, out, act_type, clip, N_total, cols)
        k = cl.Kernel(self._program, "fused_delta_act_clip_f32")
        k.set_arg(5, self._c_bxhidden2);  k.set_arg(6, self._c_hidden2)
        self._pb_dac_hid = k

        k = cl.Kernel(self._program, "fused_delta_act_clip_f32")
        k.set_arg(5, self._c_bxfull);  k.set_arg(6, self._c_full)
        self._pb_dac_full = k

        # matmul_at_b_clip_f32(A, B, C, M, N, clip)
        k = cl.Kernel(self._program, "matmul_at_b_clip_f32")
        k.set_arg(3, self._c_half);  k.set_arg(4, self._c_hidden2)
        self._pb_atb_l0 = k

        k = cl.Kernel(self._program, "matmul_at_b_clip_f32")
        k.set_arg(3, self._c_hidden2);  k.set_arg(4, self._c_full)
        self._pb_atb_l1 = k

        # sum_axis0_clip_f32(X, Y, M, N, clip)
        k = cl.Kernel(self._program, "sum_axis0_clip_f32")
        k.set_arg(2, self._c_batch);  k.set_arg(3, self._c_hidden2)
        self._pb_sa0_hid = k

        k = cl.Kernel(self._program, "sum_axis0_clip_f32")
        k.set_arg(2, self._c_batch);  k.set_arg(3, self._c_full)
        self._pb_sa0_full = k

        # fused_sgd_with_transpose_f32(W, W_T, V, G, lr, mom, decay, M, N)
        # Layer 0 weights: (half, hidden*2)
        k = cl.Kernel(self._program, "fused_sgd_with_transpose_f32")
        k.set_arg(7, self._c_half);  k.set_arg(8, self._c_hidden2)
        self._pb_sgd_l0 = k
        self._gs_sgd_l0 = (int(self.half * self.hidden * 2),)

        # Layer 1 weights: (hidden*2, full)
        k = cl.Kernel(self._program, "fused_sgd_with_transpose_f32")
        k.set_arg(7, self._c_hidden2);  k.set_arg(8, self._c_full)
        self._pb_sgd_l1 = k
        self._gs_sgd_l1 = (int(self.hidden * 2 * self.full),)

        # fused_sgd_momentum_bias_f32(B, V, G, lr, mom, N)
        k = cl.Kernel(self._program, "fused_sgd_momentum_bias_f32")
        k.set_arg(5, self._c_hidden2)
        self._pb_sgdb_hid = k

        k = cl.Kernel(self._program, "fused_sgd_momentum_bias_f32")
        k.set_arg(5, self._c_full)
        self._pb_sgdb_full = k

        # subtract_divide_loss_f32(A, B, out, partial_sums, total, D)
        total = self.batch * self.full
        ls = 256
        ng = (total + ls - 1) // ls
        k = cl.Kernel(self._program, "subtract_divide_loss_f32")
        k.set_arg(4, np.int32(total))
        self._pb_sdl = k
        self._sdl_gs = (ng * ls,)
        self._sdl_ls = (ls,)
        self._sdl_partial = cl_array.empty(self.queue, (ng,), dtype=np.float32)

        # Pre-bound soft_clip for matmul_at_b_clip CLBlast output
        k = cl.Kernel(self._program, "soft_clip_f32")
        k.set_arg(3, np.int32(self.half * self.hidden * 2))
        self._pb_atb_clip_l0 = k
        self._gs_atb_clip_l0 = (int(self.half * self.hidden * 2),)

        k = cl.Kernel(self._program, "soft_clip_f32")
        k.set_arg(3, np.int32(self.hidden * 2 * self.full))
        self._pb_atb_clip_l1 = k
        self._gs_atb_clip_l1 = (int(self.hidden * 2 * self.full),)

        # fused_coupling_grads_concat_f32(s1_outer, diff2, diff1_total, u1, st2_raw,
        #                                  diffs_st1, diffs_st2, s_limit, s_clip, t_clip, M, half_N)
        k = cl.Kernel(self._program, "fused_coupling_grads_concat_f32")
        k.set_arg(7, self._c_limit2)       # s_limit always 2.0
        # args 8 (s_clip_limit), 9 (t_clip_limit) set at call time
        k.set_arg(10, self._c_batch)
        k.set_arg(11, self._c_half)
        self._pb_fused_grads_concat = k

        # coupling_s_outer_concat_f32(diff, U, ST, combined, s_outer, s_limit, clip, M, half_N)
        k = cl.Kernel(self._program, "coupling_s_outer_concat_f32")
        k.set_arg(5, self._c_limit2)       # s_limit always 2.0
        # arg 6 (clip_limit) set at call time
        k.set_arg(7, self._c_batch)
        k.set_arg(8, self._c_half)
        self._pb_s_outer_concat = k

        # add2_clip_f32(A, B, out, clip_limit, N)
        k = cl.Kernel(self._program, "add2_clip_f32")
        k.set_arg(4, self._c_bxhalf)  # N = batch * half
        self._pb_add2_clip_half = k

        # coupling_backward_outer_concat_f32(diff, U, ST, out, s_limit, s_clip, t_clip, M, half_N)
        k = cl.Kernel(self._program, "coupling_backward_outer_concat_f32")
        k.set_arg(4, self._c_limit2)       # s_limit always 2.0
        # args 5 (s_clip_limit), 6 (t_clip_limit) set at call time
        k.set_arg(7, self._c_batch)
        k.set_arg(8, self._c_half)
        self._pb_bwd_outer_concat = k

        # coupling_backward_input_grad_merged_f32(diff, ST, du, limit, N, cols)
        k = cl.Kernel(self._program, "coupling_backward_input_grad_merged_f32")
        k.set_arg(3, self._c_limit2)
        k.set_arg(4, self._c_bxhalf)       # N = batch*half
        k.set_arg(5, self._c_full)          # cols = st_raw width
        self._pb_bwd_input_grad = k

    def _cached_f32(self, val):
        """Return a cached np.float32 for a given Python float."""
        c = self._cf.get(val)
        if c is None:
            c = np.float32(val)
            self._cf[val] = c
        return c

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
            elif n == 2*self.hidden: out = self._ring_single_hiddenx2.next()
            else: out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        elif m == self.batch:
            if n == self.half:   out = self._ring_batch_half.next()
            elif n == 2*self.hidden: out = self._ring_batch_hiddenx2.next()
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
    
    def matmul_bt(self, a, b_transposed):
        """Compute A @ B^T using CLBlast with b_transp flag."""
        ad = self.to_device(a, dtype=np.float32)
        bd = self.to_device(b_transposed, dtype=np.float32)
        m, k = ad.shape
        n, k2 = bd.shape  # bd is (N, K) because it's stored transposed
        if (m == self.batch and n == 2* self.hidden):
            out = self._ring_batch_hiddenx2.next()
        elif (m == 1 and n == self.full):
            out = self._ring_single_full.next()
        elif (m == 1 and n == 2* self.hidden):
            out = self._ring_single_hiddenx2.next()
        else:
            out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        pyclblast.gemm(
            self.queue, int(m), int(n), int(k),
            ad, bd, out,
            int(k), int(k2), int(n),   # a_ld=K, b_ld=K (row-major transposed), c_ld=N
            b_transp=True
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
                        # print(m,n)
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
                        # print(m,n)
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
    
    def permute_split(self, x, perm):
        x = self.to_device(x, dtype=np.float32)
        m, n = x.shape
        if m == self.batch and n == self.full:
            out1 = self._ring_batch_half.next()
            out2 = self._ring_batch_half.next()
            k = self._pb_permute_split
            k.set_arg(0, x.data)
            k.set_arg(1, perm.data)
            k.set_arg(2, out1.data)
            k.set_arg(3, out2.data)
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_b_full, None)
            return out1, out2
        # Fallback for inference / non-standard batch
        if m == 1 and n == self.full:
            out1 = self._ring_single_half.next()
            out2 = self._ring_single_half.next()
        else:
            out1 = cl_array.empty(self.queue, (m, n//2), dtype=np.float32)
            out2 = cl_array.empty(self.queue, (m, n//2), dtype=np.float32)
        self._permute_split_kernel(
            self.queue, (int(m), int(n)), None,
            x.data, perm.data, out1.data, out2.data,
            np.int32(m), np.int32(n), np.int32(n//2),
        )
        return out1, out2

    def coupling_forward_merged(self, u, st_raw, limit=2.0):
        u = self.to_device(u, dtype=np.float32)
        st_raw = self.to_device(st_raw, dtype=np.float32)
        m, n = u.shape
        if m == self.batch and n == self.half:
            out = self._ring_batch_half.next()
            k = self._pb_coupling_fwd_merged
            k.set_arg(0, u.data)
            k.set_arg(1, st_raw.data)
            k.set_arg(2, out.data)
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_b_half, None)
            return out
        if m == 1 and n == self.half:
            out = self._ring_single_half.next()
        else:
            out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        self._coupling_forward_merged_kernel(
            self.queue, (int(m), int(n)), None,
            u.data, st_raw.data, out.data,
            np.float32(limit), np.int32(m), np.int32(n),
        )
        return out

    def fused_forward_invperm_merged(self, v1, u2, inv_perm, st_raw, limit=2.0):
        v1 = self.to_device(v1, dtype=np.float32)
        u2 = self.to_device(u2, dtype=np.float32)
        st_raw = self.to_device(st_raw, dtype=np.float32)
        m, n = v1.shape
        perm_dev = self.to_device(inv_perm, dtype=np.int32)
        if m == self.batch and n == self.half:
            out = self._ring_batch_full.next()
            k = self._pb_fwd_invperm
            k.set_arg(0, v1.data)
            k.set_arg(1, u2.data)
            k.set_arg(2, perm_dev.data)
            k.set_arg(3, st_raw.data)
            k.set_arg(4, out.data)
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_b_full, None)
            return out
        if m == 1 and n == self.half:
            out = self._ring_single_full.next()
        else:
            out = cl_array.empty(self.queue, (m, 2*n), dtype=np.float32)
        self._fused_forward_invperm_kernel(
            self.queue, (int(m), int(2*n)), None,
            v1.data, u2.data, perm_dev.data, st_raw.data, out.data,
            np.int32(m), np.int32(n), np.float32(limit),
        )
        return out

    def concat_invperm(self, x1, x2, inv_perm):
        x1 = self.to_device(x1, dtype=np.float32)
        x2 = self.to_device(x2, dtype=np.float32)
        m, n = x1.shape
        perm_dev = self.to_device(inv_perm, dtype=np.int32)
        if m == self.batch and n == self.half:
            out = self._ring_batch_full.next()
            k = self._pb_concat_invperm
            k.set_arg(0, x1.data)
            k.set_arg(1, x2.data)
            k.set_arg(2, perm_dev.data)
            k.set_arg(3, out.data)
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_b_full, None)
            return out
        if m == 1 and n == self.half:
            out = self._ring_single_full.next()
        else:
            out = cl_array.empty(self.queue, (m, 2*n), dtype=np.float32)
        self._concat_invperm_kernel(
            self.queue, (int(m), int(2*n)), None,
            x1.data, x2.data, perm_dev.data, out.data,
            np.int32(m), np.int32(n),
        )
        return out

    def coupling_input_grad_merged(self, diff, st_raw, limit=2.0):
        dd = self.to_device(diff, dtype=np.float32)
        std = self.to_device(st_raw, dtype=np.float32)
        m, n = dd.shape
        if m == self.batch and n == self.half:
            out = self._ring_batch_half.next()
            k = self._pb_input_grad
            k.set_arg(0, dd.data)
            k.set_arg(1, std.data)
            k.set_arg(2, out.data)
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_bxhalf, None)
            return out
        out = cl_array.empty(self.queue, dd.shape, dtype=np.float32)
        total = dd.size
        cols = st_raw.shape[1]
        self._coupling_input_grad_kernel(
            self.queue, (int(total),), None,
            dd.data, std.data, out.data,
            np.float32(limit), np.int32(total), np.int32(cols),
        )
        return out

    def coupling_backward_merged(self, v, st_raw, limit=2.0):
        vd = self.to_device(v, dtype=np.float32)
        std = self.to_device(st_raw, dtype=np.float32)
        m, n = vd.shape
        if m == self.batch and n == self.half:
            out = self._ring_batch_half.next()
            k = self._pb_coupling_bwd
            k.set_arg(0, vd.data)
            k.set_arg(1, std.data)
            k.set_arg(2, out.data)
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_bxhalf, None)
            return out
        if m == 1 and n == self.half:
            out = self._ring_single_half.next()
        else:
            out = cl_array.empty(self.queue, vd.shape, dtype=np.float32)
        total = vd.size
        self._coupling_backward_kernel(
            self.queue, (int(total),), None,
            vd.data, std.data, out.data,
            np.float32(limit), np.int32(total),
        )
        return out

    def bias_act_dual(self, x, bias, act_type):
        x = self.to_device(x, dtype=np.float32)
        bias = self.to_device(bias, dtype=np.float32)
        m, n = x.shape
        if m == self.batch:
            if n == self.hidden * 2:
                out1 = self._ring_batch_hiddenx2.next()
                out2 = self._ring_batch_hiddenx2.next()
                k = self._pb_bad_hid
            elif n == self.full:
                out1 = self._ring_batch_full.next()
                out2 = self._ring_batch_full.next()
                k = self._pb_bad_full
            else:
                # Unrecognized width — use generic path below
                k = None
            if k is not None:
                k.set_arg(0, x.data)
                k.set_arg(1, bias.data)
                k.set_arg(2, out1.data)
                k.set_arg(3, out2.data)
                k.set_arg(6, self._c_act[act_type])
                cl.enqueue_nd_range_kernel(
                    self.queue, k,
                    self._gs_b_hidden2 if n == self.hidden * 2 else self._gs_b_full,
                    None,
                )
                return out1, out2
        # Generic fallback
        out1 = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        out2 = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        self._fused_bias_act_dual_kernel(
            self.queue, (int(m), int(n)), None,
            x.data, bias.data, out1.data, out2.data,
            np.int32(m), np.int32(n), np.int32(act_type),
        )
        return out1, out2

    def delta_act_clip(self, inputs, outputs, act_type, clip):
        m, n = inputs.shape
        if m == self.batch:
            if n == 2 * self.hidden:
                out = self._ring_batch_hiddenx2.next()
                k = self._pb_dac_hid
                gs = self._gs_bxhidden2
            elif n == self.full:
                # out = self._ring_batch_full.next()
                out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
                k = self._pb_dac_full
                gs = self._gs_bxfull
            else:
                k = None
            if k is not None:
                k.set_arg(0, inputs.data)
                k.set_arg(1, outputs.data)
                k.set_arg(2, out.data)
                k.set_arg(3, self._c_act[act_type])
                k.set_arg(4, self._cached_f32(clip))
                cl.enqueue_nd_range_kernel(self.queue, k, gs, None)
                return out
        # Generic fallback
        out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        self._fused_delta_act_clip_kernel(
            self.queue, (int(n*m),), None,
            inputs.data, outputs.data, out.data,
            np.int32(act_type), np.float32(clip), np.int32(n*m), np.int32(n),
        )
        return out
    
    def matmul_at_b_clip(self, a, b, clip):
        ad = self.to_device(a, dtype=np.float32)
        bd = self.to_device(b, dtype=np.float32)
        k_dim, m = ad.shape
        k2, n = bd.shape
        if k_dim != k2:
            raise ValueError(f"matmul_at_b shape mismatch: A ({k_dim},{m}) B ({k2},{n})")

        # Pick pre-allocated output buffer for known sizes
        if m == self.half and n == 2 * self.hidden:
            out = self.matmulatb_half_hidden
        elif m == 2 * self.hidden and n == self.full:
            out = self.matmulatb_hidden_full
        else:
            out = cl_array.empty(self.queue, (m, n), dtype=np.float32)

        # Primary path: CLBlast GEMM with A transposed.
        # CLBlast uses register blocking (no LDS barriers) and auto-tunes for RDNA.
        if not self._force_kernel_matmul and self._has_pyclblast and not self._disable_pyclblast:
            try:
                pyclblast.gemm(
                    self.queue,
                    int(m), int(n), int(k_dim),
                    ad, bd, out,
                    int(m),     # a_ld: A stored row-major as (K,M), stride = M
                    int(n),     # b_ld
                    int(n),     # c_ld
                    a_transp=True,
                )
                # Fuse soft_clip as a lightweight element-wise pass
                if clip > 0:
                    total = m * n
                    if m == self.half and n == 2 * self.hidden:
                        k = self._pb_atb_clip_l0
                        gs = self._gs_atb_clip_l0
                    elif m == 2 * self.hidden and n == self.full:
                        k = self._pb_atb_clip_l1
                        gs = self._gs_atb_clip_l1
                    else:
                        k = None
                    if k is not None:
                        k.set_arg(0, out.data)
                        k.set_arg(1, out.data)
                        k.set_arg(2, self._cached_f32(clip))
                        cl.enqueue_nd_range_kernel(self.queue, k, gs, None)
                    else:
                        self._soft_clip_kernel(
                            self.queue, (int(total),), None,
                            out.data, out.data,
                            np.float32(clip), np.int32(total),
                        )
                return out
            except Exception:
                self._disable_pyclblast = True

        # Fallback: tiled kernel (for when CLBlast is unavailable)
        if m == self.half and n == 2 * self.hidden:
            k = self._pb_atb_l0
            gs = self._gs_half_hid2_tiled
        elif m == 2 * self.hidden and n == self.full:
            k = self._pb_atb_l1
            gs = self._gs_hid2_full_tiled
        else:
            self._fused_matmul_at_b_clip_kernel(
                self.queue,
                (int(((m + 15) // 16) * 16), int(((n + 15) // 16) * 16)),
                (16, 16),
                ad.data, bd.data, out.data,
                np.int32(m), np.int32(n), np.float32(clip),
            )
            return out
        k.set_arg(0, ad.data)
        k.set_arg(1, bd.data)
        k.set_arg(2, out.data)
        k.set_arg(5, self._cached_f32(clip))
        cl.enqueue_nd_range_kernel(self.queue, k, gs, self._atb_local)
        return out

    def sum_axis0_clip(self, x, clip):
        m, n = x.shape
        if m == self.batch:
            if n == 2 * self.hidden:
                out = self._ring_batch_hiddenx2.next()
                k = self._pb_sa0_hid
                gs = self._gs_hidden2
            elif n == self.full:
                out = self._ring_batch_full.next()
                k = self._pb_sa0_full
                gs = self._gs_full
            else:
                k = None
            if k is not None:
                k.set_arg(0, x.data)
                k.set_arg(1, out.data)
                k.set_arg(4, self._cached_f32(clip))
                cl.enqueue_nd_range_kernel(self.queue, k, gs, None)
                return out
        out = cl_array.empty(self.queue, (1, n), dtype=np.float32)
        self._fused_sum_axis0_clip_kernel(
            self.queue, (int(n),), None,
            x.data, out.data,
            np.int32(m), np.int32(n), np.float32(clip),
        )
        return out

    def fused_sgd_update(self, weights, weights_t, velocity, gradient, lr, momentum, decay_factor):
        m, n = weights.shape
        total = m * n
        # Check for pre-bound layer-specific kernels
        if m == self.half and n == 2 * self.hidden:
            k = self._pb_sgd_l0
            gs = self._gs_sgd_l0
        elif m == 2 * self.hidden and n == self.full:
            k = self._pb_sgd_l1
            gs = self._gs_sgd_l1
        else:
            # Generic path
            self._fused_sgd_transpose_kernel(
                self.queue, (int(total),), None,
                weights.data, weights_t.data, velocity.data, gradient.data,
                np.float32(lr), np.float32(momentum), np.float32(decay_factor),
                np.int32(m), np.int32(n),
            )
            return
        k.set_arg(0, weights.data)
        k.set_arg(1, weights_t.data)
        k.set_arg(2, velocity.data)
        k.set_arg(3, gradient.data)
        k.set_arg(4, self._cached_f32(lr))
        k.set_arg(5, self._cached_f32(momentum))
        k.set_arg(6, self._cached_f32(decay_factor))
        cl.enqueue_nd_range_kernel(self.queue, k, gs, None)

    def fused_sgd_bias_update(self, bias, velocity, gradient, lr, momentum):
        n = bias.size
        if n == 2 * self.hidden:
            k = self._pb_sgdb_hid
            gs = self._gs_hidden2
        elif n == self.full:
            k = self._pb_sgdb_full
            gs = self._gs_full
        else:
            self._fused_sgd_bias_kernel(
                self.queue, (int(n),), None,
                bias.data, velocity.data, gradient.data,
                np.float32(lr), np.float32(momentum), np.int32(n),
            )
            return
        k.set_arg(0, bias.data)
        k.set_arg(1, velocity.data)
        k.set_arg(2, gradient.data)
        k.set_arg(3, self._cached_f32(lr))
        k.set_arg(4, self._cached_f32(momentum))
        cl.enqueue_nd_range_kernel(self.queue, k, gs, None)

    def subtract_divide_loss(self, a, b, d):
        a = self.to_device(a, dtype=np.float32)
        b = self.to_device(b, dtype=np.float32)
        m, n = a.shape
        if m == self.batch and n == self.full:
            out = self._ring_batch_full.next()
            ps = self._sdl_partial
            k = self._pb_sdl
            k.set_arg(0, a.data)
            k.set_arg(1, b.data)
            k.set_arg(2, out.data)
            k.set_arg(3, ps.data)
            k.set_arg(5, self._cached_f32(d))
            cl.enqueue_nd_range_kernel(self.queue, k, self._sdl_gs, self._sdl_ls)
            return out, ps
        # Generic fallback
        out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        total = m * n
        local_size = 256
        num_groups = (total + local_size - 1) // local_size
        global_size = num_groups * local_size
        partial_sums = cl_array.empty(self.queue, (num_groups,), dtype=np.float32)
        self._subtract_divide_loss_kernel(
            self.queue, (int(global_size),), (int(local_size),),
            a.data, b.data, out.data, partial_sums.data,
            np.int32(total), np.float32(d),
        )
        return out, partial_sums
    
    def fused_coupling_grads_concat(self, s1_outer, diff2, diff1_total, u1, st2_raw,
                                s_limit=2.0, s_clip_limit=1.0, t_clip_limit=0.5):
        """Fused: build diffs_st1 and diffs_st2 from coupling gradients + soft_clip + concat.
        Returns (diffs_st1, diffs_st2), each (M, 2*half_N)."""
        s1_outer = self.to_device(s1_outer, dtype=np.float32)
        diff2 = self.to_device(diff2, dtype=np.float32)
        diff1_total = self.to_device(diff1_total, dtype=np.float32)
        u1 = self.to_device(u1, dtype=np.float32)
        st2_raw = self.to_device(st2_raw, dtype=np.float32)
        m, n = s1_outer.shape  # (B, half)

        if m == self.batch and n == self.half:
            out1 = self._ring_batch_full.next()
            out2 = self._ring_batch_full.next()
            k = self._pb_fused_grads_concat
            k.set_arg(0, s1_outer.data)
            k.set_arg(1, diff2.data)
            k.set_arg(2, diff1_total.data)
            k.set_arg(3, u1.data)
            k.set_arg(4, st2_raw.data)
            k.set_arg(5, out1.data)
            k.set_arg(6, out2.data)
            k.set_arg(8, self._cached_f32(s_clip_limit))
            k.set_arg(9, self._cached_f32(t_clip_limit))
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_b_full, None)
            return out1, out2

        # Generic fallback for non-standard batch sizes
        out1 = cl_array.empty(self.queue, (m, 2 * n), dtype=np.float32)
        out2 = cl_array.empty(self.queue, (m, 2 * n), dtype=np.float32)
        self._fused_coupling_grads_concat_kernel(
            self.queue, (int(m), int(2 * n)), None,
            s1_outer.data, diff2.data, diff1_total.data, u1.data, st2_raw.data,
            out1.data, out2.data,
            np.float32(s_limit), np.float32(s_clip_limit), np.float32(t_clip_limit),
            np.int32(m), np.int32(n),
        )
        return out1, out2
    
    def coupling_s_outer_concat(self, diff, u, st_raw, s_limit=2.0, clip_limit=2.0):
        """Fused s_outer_grad + concat_cols.
        Returns (combined, s_outer) where combined = [s_outer | diff]."""
        diff = self.to_device(diff, dtype=np.float32)
        u = self.to_device(u, dtype=np.float32)
        st_raw = self.to_device(st_raw, dtype=np.float32)
        m, n = u.shape  # n = half
        if m == self.batch and n == self.half:
            out_combined = self._ring_batch_full.next()
            out_s_outer = self._ring_batch_half.next()
            k = self._pb_s_outer_concat
            k.set_arg(0, diff.data)
            k.set_arg(1, u.data)
            k.set_arg(2, st_raw.data)
            k.set_arg(3, out_combined.data)
            k.set_arg(4, out_s_outer.data)
            k.set_arg(6, self._cached_f32(clip_limit))
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_b_full, None)
            return out_combined, out_s_outer
        # Generic fallback
        out_combined = cl_array.empty(self.queue, (m, 2 * n), dtype=np.float32)
        out_s_outer = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        self._coupling_s_outer_concat_kernel(
            self.queue, (int(m), int(2 * n)), None,
            diff.data, u.data, st_raw.data,
            out_combined.data, out_s_outer.data,
            np.float32(s_limit), np.float32(clip_limit),
            np.int32(m), np.int32(n),
        )
        return out_combined, out_s_outer
    
    def add2_clip(self, a, b, clip):
        """Compute soft_clip(A + B, clip) in a single kernel."""
        ad = self.to_device(a, dtype=np.float32)
        bd = self.to_device(b, dtype=np.float32)
        m, n = ad.shape
        if m == self.batch and n == self.half:
            out = self._ring_batch_half.next()
            k = self._pb_add2_clip_half
            k.set_arg(0, ad.data)
            k.set_arg(1, bd.data)
            k.set_arg(2, out.data)
            k.set_arg(3, self._cached_f32(clip))
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_bxhalf, None)
            return out
        # Generic fallback
        total = ad.size
        out = cl_array.empty(self.queue, (m, n), dtype=np.float32)
        self._add2_clip_kernel(
            self.queue, (int(total),), None,
            ad.data, bd.data, out.data,
            np.float32(clip), np.int32(total),
        )
        return out

    def coupling_backward_outer_concat(self, diff, u, st_raw, s_limit=2.0, s_clip_limit=1.0, t_clip_limit=0.5):
        """Outer gradient for backward coupling u = (v - t) * exp(-s_clip).
        Returns combined (M, 2*half) = [s_outer | t_outer]."""
        diff = self.to_device(diff, dtype=np.float32)
        u = self.to_device(u, dtype=np.float32)
        st_raw = self.to_device(st_raw, dtype=np.float32)
        m, n = u.shape  # n = half
        if m == self.batch and n == self.half:
            out = self._ring_batch_full.next()
            k = self._pb_bwd_outer_concat
            k.set_arg(0, diff.data)
            k.set_arg(1, u.data)
            k.set_arg(2, st_raw.data)
            k.set_arg(3, out.data)
            k.set_arg(5, self._cached_f32(s_clip_limit))
            k.set_arg(6, self._cached_f32(t_clip_limit))
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_b_full, None)
            return out
        # Generic fallback
        out = cl_array.empty(self.queue, (m, 2 * n), dtype=np.float32)
        self._coupling_backward_outer_concat_kernel(
            self.queue, (int(m), int(2 * n)), None,
            diff.data, u.data, st_raw.data, out.data,
            np.float32(s_limit), np.float32(s_clip_limit), np.float32(t_clip_limit),
            np.int32(m), np.int32(n),
        )
        return out

    def coupling_backward_input_grad_merged(self, diff, st_raw, limit=2.0):
        """dv = diff * exp(-soft_clip(s_raw, limit)), where s_raw = st_raw[:, :half]."""
        dd = self.to_device(diff, dtype=np.float32)
        std = self.to_device(st_raw, dtype=np.float32)
        m, n = dd.shape
        if m == self.batch and n == self.half:
            out = self._ring_batch_half.next()
            k = self._pb_bwd_input_grad
            k.set_arg(0, dd.data)
            k.set_arg(1, std.data)
            k.set_arg(2, out.data)
            cl.enqueue_nd_range_kernel(self.queue, k, self._gs_bxhalf, None)
            return out
        out = cl_array.empty(self.queue, dd.shape, dtype=np.float32)
        total = dd.size
        cols = st_raw.shape[1]
        self._coupling_backward_input_grad_merged_kernel(
            self.queue, (int(total),), None,
            dd.data, std.data, out.data,
            np.float32(limit), np.int32(total), np.int32(cols),
        )
        return out