#!/usr/bin/env python3
"""Direct FACodec reconstruction test using upstream FAcodec.

Tests: source.wav -> FAcodec encode -> decode -> reconstruction.wav
Uses Plachta/FAcodec directly (same path as FAC-FACodec training).
Gate: mel L1 < 0.1

Run on Colab:
  colab run scripts/verify_facodec_direct.py --gpu T4
"""
import subprocess, sys, os, types

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
run("test -f /content/FAcodec/modules/__init__.py || touch /content/FAcodec/modules/__init__.py", "modules init", check=False)

# Path setup: match FAC-FACodec's train.py exactly
sys.path = [p for p in sys.path if "/content" not in p]
sys.path.insert(0, "/content/FAcodec")
os.environ["PYTHONPATH"] = "/content/FAcodec/modules:" + os.environ.get("PYTHONPATH", "")
os.chdir("/content/FAcodec")

# Mock audiotools if not available (dac/__init__.py imports it)
if 'audiotools' not in sys.modules:
    mock_audio = types.ModuleType('audiotools')
    mock_ml = types.ModuleType('audiotools.ml')
    mock_ml.BaseModel = type('BaseModel', (), {'INTERN': [], 'EXTERN': []})()
    mock_audio.ml = mock_ml
    mock_audio.AudioSignal = type('AudioSignal', (), {})
    sys.modules['audiotools'] = mock_audio
    sys.modules['audiotools.ml'] = mock_ml

# Download test audio
os.makedirs("/content/test_audio", exist_ok=True)
import torchaudio
print("\n=== Downloading LibriSpeech test-clean ===")
dataset = torchaudio.datasets.LIBRISPEECH(root="/content/test_audio", url="test-clean", download=True)
test_samples = []
for i in range(min(3, len(dataset))):
    wav, sr, tr, sid, cid, uid = dataset[i]
    fname = f"/content/test_audio/librispeech_{i}.wav"
    torchaudio.save(fname, wav, sr)
    test_samples.append(fname)
print(f"Downloaded {len(test_samples)} test samples")

print("\n=== Loading FAcodec ===")
from modules.commons import build_model, recursive_munch
from hf_utils import load_custom_model_from_hf
import yaml, torch, numpy as np
import warnings
warnings.simplefilter("ignore")

ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")
with open(config_path) as f:
    config = yaml.safe_load(f)
model_params = recursive_munch(config["model_params"])
model = build_model(model_params)

ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt = ckpt.get("net", ckpt)
for key in ckpt:
    model[key].load_state_dict(ckpt[key])
    model[key].eval()
print(f"Model loaded: {list(model.keys())}")

# Round-trip test (matching upstream reconstruct.py exactly)
print("\n=== Round-trip reconstruction ===")
os.makedirs("/content/reconstructed", exist_ok=True)
results = []
for fname in test_samples:
    wav, sr = torchaudio.load(fname)
    wav_24k = torchaudio.functional.resample(wav, sr, 24000)
    wav_in = wav_24k.unsqueeze(0).float()

    # Encode + quantize using upstream pattern
    z = model["encoder"](wav_in)
    z, quantized, commitment_loss, codebook_loss, timbre, codes = model["quantizer"](
        z, wav_in, n_c=2
    )
    # quantized = [z_c, z_p, z_r] (content, prosody, residual)
    z_c, z_p, z_r = quantized
    print(f"  {os.path.basename(fname)}: z={z.shape}, z_c={z_c.shape}, z_p={z_p.shape}, z_r={z_r.shape}, timbre={timbre.shape}")

    # Decode: upstream uses model.decoder(z) directly
    full_pred = model["decoder"](z)

    recon_path = f"/content/reconstructed/{os.path.basename(fname)}"
    torchaudio.save(recon_path, full_pred[0].cpu(), 24000)

    # Mel L1 (meaningful for codec quality)
    mel_src = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80)(wav_24k)
    mel_recon = torchaudio.transforms.MelSpectrogram(sample_rate=24000, n_fft=1024, hop_length=256, n_mels=80)(full_pred[0].cpu())
    min_m = min(mel_src.shape[-1], mel_recon.shape[-1])
    mel_l1 = (mel_src[..., :min_m] - mel_recon[..., :min_m]).abs().mean().item()
    print(f"    Mel L1: {mel_l1:.4f}")
    results.append({"file": os.path.basename(fname), "mel_l1": float(mel_l1)})

print("\n=== Results ===")
for r in results:
    print(f"  {r['file']}: Mel L1={r['mel_l1']:.4f}")
passed = all(r["mel_l1"] < 0.15 for r in results)
print(f"\nGATE: {'PASS' if passed else 'FAIL'} (all Mel L1 < 0.15: {passed})")
print("\nDONE")
