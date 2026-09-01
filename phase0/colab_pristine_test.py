#!/usr/bin/env python3
"""Pristine CosyAccent sanity test on Colab GPU."""
import subprocess
import sys

# 1. Dependencies
print("[1/4] Installing dependencies...")
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "torch", "torchaudio", "numpy", "scipy", "soundfile",
    "librosa", "hyperpyyaml", "huggingface_hub",
    "openai-whisper", "resemblyzer", "einops"
], check=True)

# 2. Clone pristine CosyAccent
print("[2/4] Cloning pristine CosyAccent...")
subprocess.run([
    "git", "clone", "-q",
    "https://github.com/P1ping/CosyAccent.git",
    "/content/cosyaccent-pristine"
], check=True)

# 3. Run paper checkpoint
print("[3/4] Running paper checkpoint...")
r = subprocess.run([
    sys.executable, "/content/cosyaccent-pristine/infer_wav.py",
    "--source_wav", "/content/source_001.wav",
    "--output_wav", "/content/test_paper_pristine.wav",
    "--model_tag", "paper",
    "--device", "cuda",
], capture_output=True, text=True)
print(r.stdout[-1000:] if r.stdout else "")
if r.returncode != 0:
    print(f"PAPER ERROR: {r.stderr[-500:]}")

# 4. Run emilia_pretrained checkpoint
print("[4/4] Running emilia_pretrained checkpoint...")
r = subprocess.run([
    sys.executable, "/content/cosyaccent-pristine/infer_wav.py",
    "--source_wav", "/content/source_001.wav",
    "--output_wav", "/content/test_emilia_pristine.wav",
    "--model_tag", "emilia_pretrained",
    "--device", "cuda",
], capture_output=True, text=True)
print(r.stdout[-1000:] if r.stdout else "")
if r.returncode != 0:
    print(f"EMILIA ERROR: {r.stderr[-500:]}")

print("\n[DONE] Both checkpoints processed.")
print("Files: /content/test_paper_pristine.wav, /content/test_emilia_pristine.wav")
