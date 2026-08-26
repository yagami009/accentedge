#!/usr/bin/env python3
"""Colab session validation script.

Run this first to verify the Colab environment before any heavy work.
Exit early if something is wrong.
"""
from __future__ import annotations

import sys
import torch
import subprocess
import json

def check(flag: bool, msg: str):
    if not flag:
        print(f"FAIL: {msg}")
        sys.exit(1)

print("=== AccentEdge Phase 1 — Environment Verification ===")

# Check CUDA
check(torch.cuda.is_available(), "CUDA is not available")
print(f"torch {torch.__version__} | CUDA {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0)}")

# Quick CUDA operation
x = torch.randn(10, 10, device="cuda")
y = x @ x.T
assert y.shape == (10, 10), "CUDA compute check failed"
print("CUDA compute verified")

print("All checks passed")
