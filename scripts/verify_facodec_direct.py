#!/usr/bin/env python3
"""Direct FACodec reconstruction test using upstream FAcodec.

Tests: source.wav -> upstream FAcodec encode -> decode -> reconstruction.wav
Uses Plachta/FAcodec directly (same path as FAC-FACodec training).

Run on Colab:
  colab run scripts/verify_facodec_direct.py --gpu T4
"""
import subprocess, sys, os

def run(cmd, desc="", check=True, timeout=120):
    print(f"\n>>> {desc or cmd[:80]}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    if out: print(out[:500])
    if check and r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}"); sys.exit(1)
    return r

run("nvidia-smi --query-gpu=name --format=csv,noheader", "GPU", check=False)
run("pip install -q numpy soundfile librosa scipy jiwer pyyaml huggingface-hub phonemizer speechbrain torchaudio faster-whisper pytest pyworld munch", "install deps")

# Clone FAcodec (skip if exists)
run("test -d /content/FAcodec || git clone https://github.com/Plachtaa/FAcodec.git /content/FAcodec", "clone FAcodec", check=False)

# Path setup
sys.path.insert(0, "/content/FAcodec")
os.environ["PYTHONPATH"] = "/content/FAcodec:" + os.environ.get("PYTHONPATH", "")
os.chdir("/content/FAcodec")

# Download test audio
os.makedirs("/content/test_audio", exist_ok=True)
import torchaudio
print("\n=== Downloading LibriSpeech test-clean ===")
dataset = torchaudio.datasets.LIBRISPEECH(root="/content/test_audio", url="test-clean", download=True)
test_samples = []
for i in range(min(3, len(dataset))):
    wav, sr, transcript, sid, cid, uid = dataset[i]
    fname = f"/content/test_audio/librispeech_{i}.wav"
    torchaudio.save(fname, wav, sr)
    test_samples.append((fname, transcript))
    print(f"  {fname}: sr={sr}, text='{transcript[:60]}...'")

print("\n=== Loading FAcodec ===")
from modules.commons import build_model, recursive_munch
from hf_utils import load_custom_model_from_hf

ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")
with open(config_path) as f:
    config = yaml.safe_load(f)

model_params = recursive_munch(config["model_params"])
model = build_model(model_params)

ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt = ckpt.get("net", ckpt)
for key in ckpt:
    model[key].load_state_dict(ckpt[key])
_ = [model[key].eval().cuda() for key in model]

print("  FAcodec loaded successfully")
print(f"  Keys: {list(model.keys())}")

# Test reconstruction
print("\n=== Testing Reconstruction ===")
os.makedirs("/content/reconstructed", exist_ok=True)
results = []

for fname, transcript in test_samples:
    wav, sr = torchaudio.load(fname)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    wav_24k = torchaudio.functional.resample(wav, sr, 24000)

    # Encode (same as upstream reconstruct.py)
    z = model["encoder"](wav_24k[None, ...].cuda().float())
    z, quantized, commit_loss, codebook_loss, timbre = model["quantizer"](z, wav_24k[None, ...].cuda().float(), n_c=2)

    # z is z_c (combined zc1+zc2)
    # Decode (same as upstream: model.decoder(z))
    recon = model["decoder"](z)

    recon_path = f"/content/reconstructed/{os.path.basename(fname)}"
    torchaudio.save(recon_path, recon[0].cpu(), 24000)

    # SNR
    src_np = wav_24k.squeeze().cpu().numpy()
    recon_np = recon[0, 0].cpu().numpy()
    min_len = min(len(src_np), len(recon_np))
    snr = 10 * np.log10(np.mean(src_np[:min_len]**2) / (np.mean((src_np[:min_len] - recon_np[:min_len])**2) + 1e-10))
    print(f"  {os.path.basename(fname)}: SNR={snr:.2f} dB")
    results.append({"file": os.path.basename(fname), "snr": float(snr)})

print("\n=== Results ===")
for r in results:
    print(f"  {r['file']}: SNR={r['snr']:.2f} dB")
passed = all(r["snr"] > 5 for r in results)
print(f"\nGATE: {'PASS' if passed else 'FAIL'} (all SNR > 5dB: {passed})")
print("\nDONE")
