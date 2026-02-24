from .numpy_backend import NumpyBackend
from .opencl_backend import OpenCLBackend


_BACKEND_SINGLETONS = {}


def get_backend(name="cpu"):
    normalized = (name or "cpu").strip().lower()
    if normalized in ("cpu", "numpy"):
        backend = _BACKEND_SINGLETONS.get("cpu")
        if backend is None:
            backend = NumpyBackend()
            _BACKEND_SINGLETONS["cpu"] = backend
        return backend
    if normalized in ("opencl", "gpu", "amd"):
        backend = _BACKEND_SINGLETONS.get("opencl")
        if backend is not None:
            return backend
        try:
            backend = OpenCLBackend(preferred_device_type="gpu")
            _BACKEND_SINGLETONS["opencl"] = backend
            return backend
        except Exception as exc:
            print(f"[Backends] OpenCL unavailable ({exc}); falling back to CPU.")
            cpu_backend = _BACKEND_SINGLETONS.get("cpu")
            if cpu_backend is None:
                cpu_backend = NumpyBackend()
                _BACKEND_SINGLETONS["cpu"] = cpu_backend
            return cpu_backend
    raise ValueError(f"Unknown backend '{name}'")
