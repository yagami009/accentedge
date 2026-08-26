#!/usr/bin/env python3
"""Colab Phase 1 bootstrap — clones repo from GitHub and runs tests."""
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

# 1. GPU check
run("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", "GPU", check=False)

# 2. Install deps (skip torch — Colab preinstalls CUDA torch)
run("pip install -q numpy soundfile librosa scipy jiwer pyyaml einops huggingface-hub phonemizer speechbrain faster-whisper pytest", "install deps")

# 3. Clone FAC-FACodec
run("git clone --depth 1 https://github.com/Claussss/FAC-FACodec.git /content/FAC-FACodec", "clone FAC-FACodec")

# 4. Clone accentedge from GitHub
run("git clone https://github.com/yagami009/accentedge.git /content/accentedge", "clone accentedge")

# 5. Set PYTHONPATH
os.environ["PYTHONPATH"] = "/content/FAC-FACodec:/content/accentedge/src"
with open("/content/colab_env.sh", "w") as f:
    f.write("export AMPHION_PATH=/content/Amphion\n")
    f.write("export PYTHONPATH=/content/FAC-FACodec:/content/accentedge/src:$PYTHONPATH\n")

# 6. Run Phase 1 tests
os.chdir("/content/accentedge")
print("\n=== Running Phase 1 tests ===")
r = subprocess.run(
    ["python3", "-m", "pytest", "tests/test_phase1.py", "-v", "--tb=short"],
    capture_output=True, text=True, timeout=120
)
print(r.stdout[-3000:])
if r.stderr:
    print("STDERR:", r.stderr[-2000:])
sys.exit(r.returncode)
