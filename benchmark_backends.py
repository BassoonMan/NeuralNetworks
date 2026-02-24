import random
import time
import numpy as np
import sys
import argparse

sys.path.insert(0, r"c:\Users\joshu\Documents\AI\NeuralNetworks\InvertibleNN")
sys.path.insert(0, r"c:\Users\joshu\Documents\AI\NeuralNetworks")
from INN import InvertibleNeuralNetwork
from Backends.opencl_backend import OpenCLBackend


def run_backend(backend: str):
    np.random.seed(42)
    random.seed(42)
    inn = InvertibleNeuralNetwork(5, 4, 2, 16, backend=backend)

    x = np.atleast_2d([1, 0, 0, 1]).astype(np.float32)
    y = np.atleast_2d([0.9, 0.9, 0.1, 0.1]).astype(np.float32)

    for _ in range(20):
        out = inn.forward(x)
        _ = inn.backward(out)
        out_t = inn.train_forward(x)
        inn.backpropagate(y - out_t, 0.001)

    n_fwd = 600
    t0 = time.perf_counter()
    for _ in range(n_fwd):
        _ = inn.forward(x)
    t_fwd = (time.perf_counter() - t0) / n_fwd

    n_bwd = 600
    out = inn.forward(x)
    t0 = time.perf_counter()
    for _ in range(n_bwd):
        _ = inn.backward(out)
    t_bwd = (time.perf_counter() - t0) / n_bwd

    n_tr = 120
    t0 = time.perf_counter()
    for _ in range(n_tr):
        out_t = inn.train_forward(x)
        inn.backpropagate(y - out_t, 0.001)
    t_train = (time.perf_counter() - t0) / n_tr

    return t_fwd, t_bwd, t_train


def benchmark_batch_modes(backend: str, total_rows: int, chunk_size: int, repeats: int):
    np.random.seed(42)
    random.seed(42)
    inn = InvertibleNeuralNetwork(5, 4, 2, 16, backend=backend)

    x = np.random.rand(total_rows, 4).astype(np.float32)
    y = np.random.rand(total_rows, 4).astype(np.float32)
    loop_rows = min(total_rows, 128)
    x_loop = x[:loop_rows]
    y_loop = y[:loop_rows]

    # Warmup
    _ = inn.forward_batch(x)
    _ = inn.forward_batch(x, batch_size=chunk_size)
    _ = np.vstack([inn.forward(np.atleast_2d(row)) for row in x_loop])

    _ = inn.backward_batch(y)
    _ = inn.backward_batch(y, batch_size=chunk_size)
    _ = np.vstack([inn.backward(np.atleast_2d(row)) for row in y_loop])

    # Forward modes
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = inn.forward_batch(x)
    t_forward_full = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = inn.forward_batch(x, batch_size=chunk_size)
    t_forward_chunked = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = np.vstack([inn.forward(np.atleast_2d(row)) for row in x_loop])
    t_forward_loop_subset = (time.perf_counter() - t0) / repeats
    t_forward_loop = t_forward_loop_subset * (total_rows / loop_rows)

    # Backward modes
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = inn.backward_batch(y)
    t_backward_full = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = inn.backward_batch(y, batch_size=chunk_size)
    t_backward_chunked = (time.perf_counter() - t0) / repeats

    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = np.vstack([inn.backward(np.atleast_2d(row)) for row in y_loop])
    t_backward_loop_subset = (time.perf_counter() - t0) / repeats
    t_backward_loop = t_backward_loop_subset * (total_rows / loop_rows)

    return {
        "forward": {
            "full_batch": t_forward_full,
            "chunked": t_forward_chunked,
            "sample_loop": t_forward_loop,
        },
        "backward": {
            "full_batch": t_backward_full,
            "chunked": t_backward_chunked,
            "sample_loop": t_backward_loop,
        },
    }


def benchmark_matmul(m: int, n: int, k: int, repeats: int):
    np.random.seed(42)
    a = np.random.randn(m, k).astype(np.float32)
    b = np.random.randn(k, n).astype(np.float32)

    # CPU NumPy baseline
    _ = a @ b
    t0 = time.perf_counter()
    for _ in range(repeats):
        _ = a @ b
    t_cpu = (time.perf_counter() - t0) / repeats

    clb = OpenCLBackend()

    # OpenCL + CLBlast path
    clb.set_matmul_engine("clblast")
    out_clblast = clb.matmul(a, b)
    clb.queue.finish()

    t0 = time.perf_counter()
    for _ in range(repeats):
        out_clblast = clb.matmul(a, b)
    clb.queue.finish()
    t_clblast = (time.perf_counter() - t0) / repeats

    c_ref = a @ b
    c_clblast = clb.to_host(out_clblast)
    err_clblast = float(np.max(np.abs(c_clblast - c_ref)))

    # OpenCL fallback kernel path
    clb.set_matmul_engine("kernel")
    out_kernel = clb.matmul(a, b)
    clb.queue.finish()

    t0 = time.perf_counter()
    for _ in range(repeats):
        out_kernel = clb.matmul(a, b)
    clb.queue.finish()
    t_kernel = (time.perf_counter() - t0) / repeats

    c_kernel = clb.to_host(out_kernel)
    err_kernel = float(np.max(np.abs(c_kernel - c_ref)))

    return {
        "shape": (m, n, k),
        "seconds_per_call": {
            "cpu_numpy": t_cpu,
            "opencl_clblast": t_clblast,
            "opencl_kernel": t_kernel,
        },
        "speedup_vs_cpu": {
            "opencl_clblast": (t_cpu / t_clblast) if t_clblast > 0 else float("inf"),
            "opencl_kernel": (t_cpu / t_kernel) if t_kernel > 0 else float("inf"),
        },
        "accuracy": {
            "max_abs_err_clblast": err_clblast,
            "max_abs_err_kernel": err_kernel,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["basic", "batch", "matmul"], default="basic")
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=40)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=512)
    parser.add_argument("--k", type=int, default=512)
    args = parser.parse_args()

    if args.mode == "basic":
        cpu = run_backend("cpu")
        gpu = run_backend("opencl")

        print("per_call_seconds")
        print("cpu", "forward", cpu[0], "backward", cpu[1], "train_step", cpu[2])
        print("opencl", "forward", gpu[0], "backward", gpu[1], "train_step", gpu[2])

        print("speedup_opencl_vs_cpu")
        print("forward", (cpu[0] / gpu[0]) if gpu[0] > 0 else float("inf"))
        print("backward", (cpu[1] / gpu[1]) if gpu[1] > 0 else float("inf"))
        print("train_step", (cpu[2] / gpu[2]) if gpu[2] > 0 else float("inf"))
        return

    if args.mode == "matmul":
        result = benchmark_matmul(args.m, args.n, args.k, args.repeats)
        print("matmul_benchmark")
        print("shape_m_n_k", result["shape"])
        print("seconds_per_call", result["seconds_per_call"])
        print("speedup_vs_cpu", result["speedup_vs_cpu"])
        print("accuracy", result["accuracy"])
        return

    cpu = benchmark_batch_modes("cpu", args.rows, args.chunk, args.repeats)
    gpu = benchmark_batch_modes("opencl", args.rows, args.chunk, args.repeats)

    print("batch_mode_seconds")
    print("rows", args.rows, "chunk", args.chunk, "repeats", args.repeats)
    for direction in ("forward", "backward"):
        print("cpu", direction, cpu[direction])
        print("opencl", direction, gpu[direction])

    print("batch_mode_speedup_opencl_vs_cpu")
    for direction in ("forward", "backward"):
        for mode in ("full_batch", "chunked", "sample_loop"):
            c = cpu[direction][mode]
            g = gpu[direction][mode]
            print(direction, mode, (c / g) if g > 0 else float("inf"))


if __name__ == "__main__":
    main()
