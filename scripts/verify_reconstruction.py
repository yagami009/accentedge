#!/usr/bin/env python3
"""Reconstruction verification for FACodec.

Tests: source.wav -> encode -> decode -> reconstruction.wav
and computes SNR.

Run on Colab:
  colab run scripts/verify_reconstruction.py --gpu T4
"""
import subprocess, sys, os

SRC = "/content/accentedge/src"

def run(cmd, desc="", check=True, timeout=120):
    print(f"\n>>> {desc or cmd[:80]}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    if out: print(out[:500])
    if check and r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}"); sys.exit(1)
    return r

# 1. GPU
run("nvidia-smi --query-gpu=name --format=csv,noheader", "GPU", check=False)

# 2. Deps (skip torch - Colab has it)
run("pip install -q numpy soundfile librosa scipy jiwer pyyaml einops huggingface-hub phonemizer speechbrain faster-whisper pytest", "install deps")

# 3. Clone repos (skip if exist)
run("test -d /content/FAC-FACodec || git clone --depth 1 https://github.com/Claussss/FAC-FACodec.git /content/FAC-FACodec", "clone FAC-FACodec", check=False)
run("test -d /content/Amphion || git clone --depth 1 https://github.com/open-mmlab/Amphion.git /content/Amphion", "clone Amphion", check=False)
run("test -d /content/accentedge || git clone https://github.com/yagami009/accentedge.git /content/accentedge", "clone accentedge", check=False)

os.environ["PYTHONPATH"] = "/content/FAC-FACodec:/content/Amphion:/content/accentedge/src:" + os.environ.get("PYTHONPATH", "")
os.chdir("/content/accentedge")
sys.path.insert(0, SRC)

# 8. Download test audio
os.makedirs("/content/test_audio", exist_ok=True)
import torchaudio
print("\n=== Downloading LibriSpeech test-clean ===")
try:
    dataset = torchaudio.datasets.LIBRISPEECH(root="/content/test_audio", url="test-clean", download=True)
    test_samples = []
    for i in range(min(3, len(dataset))):
        wav, sr, transcript, sid, cid, uid = dataset[i]
        fname = f"/content/test_audio/librispeech_{i}.wav"
        torchaudio.save(fname, wav, sr)
        test_samples.append((fname, transcript))
        print(f"  {fname}: {wav.shape}, sr={sr}, text='{transcript[:60]}'")
except Exception as e:
    print(f"LibriSpeech download failed: {e}")
    import numpy as np
    sr = 24000
    t = np.linspace(0, 3.0, int(sr * 3.0))
    signal = 0.3 * np.sin(2 * np.pi * 150 * t) + 0.2 * np.sin(2 * np.pi * 300 * t)
    signal = signal / np.max(np.abs(signal)) * 0.5
    torchaudio.save("/content/test_audio/synthetic_1.wav", torch.tensor(signal).unsqueeze(0), sr)
    test_samples = [("/content/test_audio/synthetic_1.wav", "synthetic")]

# 9. Run reconstruction
print("\n=== FACodec Reconstruction Verification ===")
from accentedge.codec.facodec import FACodecAdapter
import torch
import soundfile as sf
import numpy as np

DEVICE = "cuda"
print(f"\nLoading FACodecAdapter on {DEVICE}...")
adapter = FACodecAdapter(device=DEVICE)

os.makedirs("/content/reconstructed", exist_ok=True)
results = []

for fname, transcript in test_samples:
    print(f"\n--- Processing {os.path.basename(fname)} ---")
    wav_np, sr = sf.read(fname, dtype="float32")
    if wav_np.ndim > 1:
        wav_np = wav_np.mean(axis=1)
    wav = torch.tensor(wav_np, dtype=torch.float32).unsqueeze(0)

    # Resample to 24kHz if needed
    if sr != 24000:
        import torchaudio.functional as F
        wav = F.resample(wav, sr, 24000)
        sr = 24000

    # Encode
    latents = adapter.encode(wav)
    print(f"  zc1: {latents.content_zc1.shape}, prosody: {latents.prosody.shape if latents.prosody is not None else 'None'}, timbre: {latents.timbre.shape if latents.timbre is not None else 'None'}")

    # Decode (round-trip)
    recon = adapter.decode(latents)
    recon_path = f"/content/reconstructed/{os.path.basename(fname)}"
    sf.write(recon_path, recon.numpy().squeeze(), sr)
    print(f"  Saved: {recon_path}")

    # SNR
    src_np = wav.numpy().squeeze()
    recon_np = recon.numpy().squeeze()
    min_len = min(len(src_np), len(recon_np))
    snr = 10 * np.log10(np.mean(src_np[:min_len]**2) / (np.mean((src_np[:min_len] - recon_np[:min_len])**2) + 1e-10))
    print(f"  SNR: {snr:.2f} dB")
    results.append({"file": os.path.basename(fname), "snr": float(snr)})

print("\n=== Results ===")
for r in results:
    print(f"  {r['file']}: SNR={r['snr']:.2f} dB")
print(f"\nAll reconstructions complete: {len(results)} files")
print("\nGATE:", "PASS" if all(r['snr'] > 10 for r in results) else "FAIL - SNR too low")
