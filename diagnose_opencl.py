import os
import sys
import platform
from importlib import metadata


def get_package_version(name: str):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main():
    print_header("Python Environment")
    print(f"Executable: {sys.executable}")
    print(f"Version   : {sys.version.splitlines()[0]}")
    print(f"Platform  : {platform.platform()}")
    print(f"CWD       : {os.getcwd()}")

    print_header("Key Package Versions")
    for pkg in ["pyopencl", "numpy", "typing_extensions"]:
        ver = get_package_version(pkg)
        print(f"{pkg:18}: {ver if ver else 'NOT INSTALLED'}")

    print_header("PyOpenCL Import")
    try:
        import pyopencl as cl
        import pyopencl.array as cl_array
        import pyopencl.clmath as clmath
        print(f"pyopencl version : {cl.__version__}")
    except Exception as exc:
        print(f"FAILED to import pyopencl: {exc}")
        print("\nDiagnosis: pyopencl is missing or has dependency/version conflicts in this interpreter.")
        return

    print_header("OpenCL Platforms and Devices")
    try:
        platforms = cl.get_platforms()
        if not platforms:
            print("No OpenCL platforms found.")
            print("Diagnosis: GPU driver/OpenCL runtime is not visible to this Python environment.")
            return

        for pi, plat in enumerate(platforms):
            print(f"[{pi}] Platform: {plat.name}")
            print(f"    Vendor : {plat.vendor}")
            print(f"    Version: {plat.version}")

            devices = plat.get_devices()
            if not devices:
                print("    No devices on this platform.")
                continue

            for di, dev in enumerate(devices):
                print(f"    [{di}] Device: {dev.name}")
                print(f"         Type      : {cl.device_type.to_string(dev.type)}")
                print(f"         Vendor    : {dev.vendor}")
                print(f"         Driver    : {dev.driver_version}")
                print(f"         Compute U : {dev.max_compute_units}")
                print(f"         Global Mem: {dev.global_mem_size // (1024 ** 2)} MB")
    except Exception as exc:
        print(f"FAILED platform/device query: {exc}")
        return

    print_header("OpenCL Compute Smoke Test")
    try:
        gpu_device = None
        for plat in platforms:
            gpus = plat.get_devices(device_type=cl.device_type.GPU)
            if gpus:
                gpu_device = gpus[0]
                break

        device = gpu_device
        if device is None:
            for plat in platforms:
                all_devices = plat.get_devices()
                if all_devices:
                    device = all_devices[0]
                    break

        if device is None:
            print("No usable OpenCL device found for compute test.")
            return

        ctx = cl.Context([device])
        queue = cl.CommandQueue(ctx)

        import numpy as np
        x_host = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]], dtype=np.float32)
        x_dev = cl_array.to_device(queue, x_host)

        alpha = 0.01
        y_dev = 0.5 * ((1.0 + alpha) * x_dev + (1.0 - alpha) * clmath.fabs(x_dev))
        y_host = y_dev.get()

        expected = np.where(x_host < 0, alpha * x_host, x_host)
        max_abs_err = float(np.max(np.abs(y_host - expected)))

        print(f"Selected device : {device.name}")
        print(f"Input           : {x_host.tolist()}")
        print(f"Output          : {y_host.tolist()}")
        print(f"Expected        : {expected.tolist()}")
        print(f"Max abs error   : {max_abs_err:.8f}")

        if max_abs_err < 1e-5:
            print("Result: PASS")
        else:
            print("Result: WARNING (numerical mismatch above threshold)")

    except Exception as exc:
        print(f"FAILED compute smoke test: {exc}")


if __name__ == "__main__":
    main()
