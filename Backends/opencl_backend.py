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

    def __init__(self, preferred_device_type="gpu"):
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
        self._program = cl.Program(self.context, self._GEMM_SOURCE).build()
        self._matmul_kernel = cl.Kernel(self._program, "matmul_f32")
        self._add_row_broadcast_kernel = cl.Kernel(self._program, "add_row_broadcast_f32")
        self._has_pyclblast = pyclblast is not None
        # If CLBlast fails once on this runtime/device, disable and fallback to
        # custom kernel to avoid repeated exception overhead.
        self._disable_pyclblast = False
        self._force_kernel_matmul = False

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

    def add(self, a, b):
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

    def subtract(self, a, b):
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

    def apply_activation(self, x, func_name):
        xd = self.to_device(x)
        if func_name == "leakyReLU":
            alpha = 0.01
            # Branchless leaky ReLU: 0.5 * ((1+alpha)x + (1-alpha)|x|)
            return 0.5 * ((1.0 + alpha) * xd + (1.0 - alpha) * clmath.fabs(xd))
        if func_name == "tanh":
            return clmath.tanh(xd)
        if func_name == "identity":
            return xd
        return None

    def apply_activation_derivative(self, x, func_name):
        xd = self.to_device(x)
        if func_name == "leakyReLU":
            alpha = 0.01
            eps = 1e-6
            sign_x = xd / (clmath.fabs(xd) + eps)
            return 0.5 * ((1.0 + alpha) + (1.0 - alpha) * sign_x)
        if func_name == "tanh":
            t = clmath.tanh(xd)
            return 1.0 - t * t
        if func_name == "identity":
            return xd * 0.0 + 1.0
        return None
