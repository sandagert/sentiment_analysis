#!/usr/bin/env python3

import torch
import sys

print("=" * 80)
print("GPU DETECTION TEST")
print("=" * 80)
print()

# Check CUDA availability
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("Number of GPUs:", torch.cuda.device_count())
    print()

    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  VRAM: {props.total_memory / 1e9:.1f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
        print()

    # Quick test
    print("Running quick GPU test...")
    x = torch.randn(1000, 1000).cuda()
    y = torch.randn(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("✓ GPU computation successful!")

else:
    print("⚠️  NO GPU DETECTED")
    print("Scripts will run on CPU (very slow)")
    print()
    print("To use GPU, ensure:")
    print("  1. NVIDIA GPU is available")
    print("  2. CUDA drivers are installed")
    print("  3. PyTorch with CUDA is installed")

print("=" * 80)