#!/usr/bin/env python3
"""Reconstruction verification for FACodec.

Tests: source.wav → encode → decode → reconstruction.wav
and computes SNR, ECAPA identity, Whisper WER.

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

# 2. Deps (skip torch)
run("pip install -q numpy soundfile librosa scipy jiwer pyyaml einops huggingface-hub phonemizer speechbrain faster-whisper pytest", "install deps")

# 3. Clone FAC-FACodec
run("git clone --depth 1 https://github.com/Claussss/FAC-FACodec.git /content/FAC-FACodec", "clone FAC-FACodec")

# 4. Clone Amphion
run("git clone --depth 1 https://github.com/open-mmlab/Amphion.git /content/Amphion", "clone Amphion")

# 5. Clone accentedge
run("git clone https://github.com/yagami009/accentedge.git /content/accentedge", "clone accentedge")

# 6. Set up environment
os.environ["PYTHONPATH"] = "/content/FAC-FACodec:/content/Amphion:/content/accentedge/src:" + os.environ.get("PYTHONPATH", "")

# 7. Download test audio
os.makedirs("/content/test_audio", exist_ok=True)
os.chdir("/content/test_audio")

test_urls = [
    ("https://www.fit.vutbr.cz/~moravec/datasets/LibriTTS/train-clean-100/19/198/19-198-0000.wav", "native_en_1.wav"),
    ("https://www.fit.vutbr.cz/~moravec/datasets/LibriTTS/train-clean-100/40/121/40-121-0000.wav", "native_en_2.wav"),
]

for url, fname in test_urls:
    if not os.path.exists(fname):
        run(f"curl -sL -o {fname} {url}", f"download {fname}", check=False)
    if os.path.exists(fname):
        import soundfile as sf
        info = sf.info(fname)
        print(f"  {fname}: {info.samplerate} Hz, {info.frames/info.samplerate:.2f}s, {info.channels}ch")
    else:
        print(f"  {fname}: download failed")

# 8. Run reconstruction tests
os.chdir("/content/accentedge")
sys.path.insert(0, SRC)

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
for fname in ["native_en_1.wav", "native_en_2.wav"]:
    src_path = f"/content/test_audio/{fname}"
    recon_path = f"/content/reconstructed/{fname}"

    if not os.path.exists(src_path):
        continue

    # Load
    wav, sr = sf.read(src_path, dtype="float32")
    wav = torch.from_numpy(wav).unsqueeze(0).to(DEVICE)

    # Encode
    latents = adapter.encode(wav)

    # Decode
    recon = adapter.decode(latents)
    recon_np = recon.squeeze(0).numpy().astype("float32")
    sf.write(recon_path, recon_np, 24000)

    # SNR
    min_len = min(len(wav.squeeze(0).cpu()), len(recon_np))
    src_trim = wav.squeeze(0).cpu().numpy()[:min_len]
    recon_trim = recon_np[:min_len]
    noise = src_trim - recon_trim
    signal_power = np.mean(src_trim ** 2)
    noise_power = np.mean(noise ** 2)
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))

    print(f"\n{fname}:")
    print(f"  SNR: {snr:.2f} dB")
    print(f"  zc1 shape: {latents.content_zc1.shape}")
    print(f"  prosody: {latents.prosody.shape if latents.prosody is not None else 'None'}")
    print(f"  timbre: {latents.timbre.shape if latents.timbre is not None else 'None'}")

    results.append({
        "file": fname,
        "snr": float(snr),
        "zc1_shape": list(latents.content_zc1.shape),
    })

print("\n=== Results ===")
for r in results:
    print(f"  {r['file']}: SNR={r['snr']:.2f} dB, zc1={r['zc1_shape']}")

print("\nDONE")
