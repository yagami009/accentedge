#!/usr/bin/env python3
"""Colab Phase 1 bootstrap — clones repo and runs tests."""
import subprocess, sys, os

SRC = "/content/accentedge/src/accentedge"

def run(cmd, desc="", check=True, timeout=120):
    print(f"\n>>> {desc or cmd[:80]}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    if out:
        print(out[:500])
    if check and r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}")
        sys.exit(1)
    return r

# Check GPU
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "GPU check", check=False)

# Install deps (skip torch — Colab has it)
run("pip install -q numpy soundfile librosa scipy jiwer pyyaml einops huggingface-hub phonemizer speechbrain faster-whisper pytest", "install deps")

# Clone FAC-FACodec
run("git clone --depth 1 https://github.com/Claussss/FAC-FACodec.git /content/FAC-FACodec", "clone FAC-FACodec")

# Clone accentedge from GitHub
run("git clone https://github.com/yagami009/accentedge.git /content/accentedge", "clone accentedge")

# Set PYTHONPATH
os.environ["PYTHONPATH"] = "/content/FAC-FACodec:/content/accentedge/src:" + os.environ.get("PYTHONPATH", "")
with open("/content/colab_env.sh", "w") as f:
    f.write("export AMPHION_PATH=/content/Amphion\n")
    f.write("export PYTHONPATH=/content/FAC-FACodec:/content/accentedge/src:$PYTHONPATH\n")

# Run Phase 1 tests
os.chdir("/content/accentedge")
r = run("python3 -m pytest tests/test_phase1.py -v --tb=short 2>&1", "run Phase 1 tests", timeout=120)

print("\n=== Bootstrap complete ===")
