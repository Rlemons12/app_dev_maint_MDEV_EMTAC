"""
Standalone GPU / accelerator test.

This script has NO project dependencies.
It answers one question:

    "What accelerators can THIS Python interpreter actually use?"

Safe to run anywhere:
- Laptop
- Server
- PyCharm
- CI
- Container
"""

import time
import platform
import torch


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def test_cuda():
    print_header("CUDA (NVIDIA) DETECTION")

    if not torch.cuda.is_available():
        print("CUDA NOT available in this Python environment.")
        return False

    count = torch.cuda.device_count()
    print(f"CUDA available: YES")
    print(f"CUDA device count: {count}")

    for i in range(count):
        props = torch.cuda.get_device_properties(i)
        total_gb = props.total_memory / (1024 ** 3)
        cap = f"{props.major}.{props.minor}"

        print(f"\nDevice {i}:")
        print(f"  Name:              {props.name}")
        print(f"  Total VRAM:        {total_gb:.2f} GB")
        print(f"  Compute capability:{cap}")

    # Simple compute test
    print("\nRunning CUDA compute test...")
    a = torch.randn((1024, 1024), device="cuda")
    b = torch.randn((1024, 1024), device="cuda")

    start = time.time()
    c = a @ b
    torch.cuda.synchronize()
    elapsed = time.time() - start

    print(f"CUDA compute OK | time={elapsed:.4f}s | result device={c.device}")
    return True


def test_mps():
    print_header("MPS (APPLE SILICON) DETECTION")

    if not hasattr(torch.backends, "mps"):
        print("MPS backend not present in this PyTorch build.")
        return False

    if not torch.backends.mps.is_available():
        print("MPS backend present but NOT available.")
        return False

    print("MPS available: YES")

    a = torch.randn((1024, 1024), device="mps")
    b = torch.randn((1024, 1024), device="mps")

    start = time.time()
    c = a @ b
    elapsed = time.time() - start

    print(f"MPS compute OK | time={elapsed:.4f}s | result device={c.device}")
    return True


def test_cpu():
    print_header("CPU FALLBACK")

    cpu_name = platform.processor() or platform.machine() or "CPU"
    print(f"CPU: {cpu_name}")

    a = torch.randn((1024, 1024))
    b = torch.randn((1024, 1024))

    start = time.time()
    c = a @ b
    elapsed = time.time() - start

    print(f"CPU compute OK | time={elapsed:.4f}s | result device={c.device}")
    return True


def main():
    print_header("STANDALONE ACCELERATOR TEST")
    print(f"Python executable: {torch.__file__}")
    print(f"PyTorch version:   {torch.__version__}")
    print(f"Platform:          {platform.platform()}")

    used_accelerator = False

    if test_cuda():
        used_accelerator = True
    elif test_mps():
        used_accelerator = True
    else:
        test_cpu()

    print_header("TEST COMPLETE")

    if used_accelerator:
        print("An accelerator is AVAILABLE and USABLE in this environment.")
    else:
        print("No GPU accelerator usable. CPU-only environment.")


if __name__ == "__main__":
    main()
