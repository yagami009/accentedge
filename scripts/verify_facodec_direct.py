#!/usr/bin/env python3
"""Direct FACodec reconstruction test using FAC-FACodec pattern."""
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
run("test -d /content/FAcodec || git clone https://github.com/Plachtaa/FAcodec.git /content/FAcodec", "clone FAcodec", check=False)
run("test -f /content/FAcodec/modules/__init__.py || touch /content/FAcodec/modules/__init__.py", "modules init", check=False)

sys.path = [p for p in sys.path if "/content" not in p]
sys.path.insert(0, "/content/FAcodec")
os.environ["PYTHONPATH"] = "/content/FAcodec:" + os.environ.get("PYTHONPATH", "")
os.chdir("/content/FAcodec")

os.makedirs("/content/test_audio", exist_ok=True)
import torchaudio
print("\n=== Downloading LibriSpeech ===")
dataset = torchaudio.datasets.LIBRISPEECH(root="/content/test_audio", url="test-clean", download=True)
test_samples = []
for i in range(min(3, len(dataset))):
    wav, sr, tr, sid, cid, uid = dataset[i]
    fname = f"/content/test_audio/librispeech_{i}.wav"
    torchaudio.save(fname, wav, sr)
    test_samples.append(fname)
    print(f"  {fname}: sr={sr}")

print("\n=== Loading FAcodec ===")
from modules.commons import build_model, recursive_munch
from hf_utils import load_custom_model_from_hf
import yaml, torch, numpy as np, soundfile as sf
import warnings
warnings.simplefilter("ignore")

ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")
with open(config_path) as f:
    config = yaml.safe_load(f)
model_params = recursive_munch(config.get("model_params", config.get("model", {})))
model = build_model(model_params)
ckpt = torch.load(ckpt_path, map_location="cpu")
for key in ckpt:
    model[key].load_state_dict(ckpt[key])
    model[key].eval()
print(f"Model keys: {list(model.keys())}")

# Round-trip test
print("\n=== Round-trip reconstruction ===")
os.makedirs("/content/reconstructed", exist_ok=True)
results = []
for fname in test_samples:
    wav, sr = torchaudio.load(fname)
    wav = wav.mean(dim=0, keepdim=True)
    wav_24k = torchaudio.functional.resample(wav, sr, 24000)
    wav_in = wav_24k.unsqueeze(0)

    # Encode
    z_c, z_p, z_t, z_r = model["quantizer"](wav_in.float(), wav_in.float(), n_c=2)
    print(f"  {os.path.basename(fname)}: z_c={z_c.shape}, z_p={z_p.shape}, z_r={z_r.shape}")

    # Decode: upstream uses z = z_c + z_p + z_r, then decoder()
    z_total = z_c.detach() + z_p.detach() + z_r.detach()
    recon = model["decoder"](z_total)

    recon_path = f"/content/reconstructed/{os.path.basename(fname)}"
    torchaudio.save(recon_path, recon[0].cpu(), 24000)

    # SNR
    src_np = wav_24k.squeeze().cpu().numpy()
    recon_np = recon[0, 0].cpu().numpy()
    min_len = min(len(src_np), len(recon_np))
    snr = 10 * np.log10(np.mean(src_np[:min_len]**2) / (np.mean((src_np[:min_len] - recon_np[:min_len])**2) + 1e-10))
    print(f"    SNR: {snr:.2f} dB")
    results.append({"file": os.path.basename(fname), "snr": float(snr)})

print("\n=== Results ===")
for r in results:
    print(f"  {r['file']}: SNR={r['snr']:.2f} dB")
passed = all(r["snr"] > 5 for r in results)
print(f"\nGATE: {'PASS' if passed else 'FAIL'} (all SNR > 5dB: {passed})")
print("\nDONE")
