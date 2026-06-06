"""Check that PyTorch sees the GPU. Exits non-zero if CUDA is missing.

    python src/check_gpu.py
"""

import sys


def main() -> int:
    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is not installed in this environment.")
        print("Fix: activate the venv and install torch from the CUDA wheel index:")
        print("  pip install torch --index-url https://download.pytorch.org/whl/cu124")
        return 1

    print("=" * 60)
    print("GPU / CUDA CHECK")
    print("=" * 60)
    print(f"PyTorch version      : {torch.__version__}")
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA available       : {cuda_ok}")

    if not cuda_ok:
        print("-" * 60)
        print("STOP: CUDA is NOT available. Training would run on the CPU.")
        print("Most common causes and fixes:")
        print("  1) A CPU-only PyTorch got installed. Reinstall the CUDA build:")
        print("       pip uninstall -y torch")
        print("       pip install torch --index-url https://download.pytorch.org/whl/cu124")
        print("  2) NVIDIA driver missing/outdated. Check with: nvidia-smi")
        print("  3) You are not inside the project's .venv.")
        print("-" * 60)
        return 1

    # CUDA is available - print the device details.
    device_index = torch.cuda.current_device()
    print(f"CUDA runtime (torch) : {torch.version.cuda}")
    print(f"Device count         : {torch.cuda.device_count()}")
    print(f"Active device index  : {device_index}")
    print(f"Device name          : {torch.cuda.get_device_name(device_index)}")
    total_mem_gb = torch.cuda.get_device_properties(device_index).total_memory / (1024**3)
    print(f"Total VRAM           : {total_mem_gb:.1f} GB")

    # Tiny real computation on the GPU to prove it works end-to-end.
    x = torch.rand(1000, 1000, device="cuda")
    y = (x @ x).sum().item()
    print(f"GPU matmul self-test : OK (result={y:.1f})")
    print("=" * 60)
    print("RESULT: GPU is detected and usable. You can proceed to training.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
