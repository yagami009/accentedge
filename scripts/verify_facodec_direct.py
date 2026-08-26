#!/usr/bin/env python3
"""FACodec reconstruction verification using Amphion.

Tests: source.wav -> Amphion FACodec encode -> decode -> reconstruction.wav
Gate: all SNR > 5dB

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

# Amphion path
sys.path.insert(0, "/content/Amphion")
os.environ["PYTHONPATH"] = "/content/Amphion:" + os.environ.get("PYTHONPATH", "")
os.chdir("/content/Amphion")

# Download test audio
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

print("\n=== Loading Amphion FACodec ===")
from models.codec.ns3_codec.facodec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
import yaml, torch, numpy as np

ckpt_path = hf_hub_download(repo_id="Plachta/FAcodec", filename="pytorch_model.bin")
config_path = hf_hub_download(repo_id="Plachta/FAcodec", filename="config.yml")

with open(config_path) as f:
    config = yaml.safe_load(f)

# Build Amphion FACodec with same config as Plachta/FAcodec
encoder = FACodecEncoder(ngf=32, up_ratios=[2,4,5,5], out_channels=256)
decoder = FACodecDecoder(
    in_channels=256, upsample_initial_channel=1024, ngf=32,
    up_ratios=[5,5,4,2],
    vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
    vq_dim=256, codebook_dim=8,
    codebook_size_prosody=10, codebook_size_content=10, codebook_size_residual=10,
    use_gr_x_timbre=True,
)

model = {"encoder": encoder, "decoder": decoder}

# Load checkpoint
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt = ckpt.get("net", ckpt)
if "encoder" in ckpt:
    model["encoder"].load_state_dict(ckpt["encoder"])
if "decoder" in ckpt:
    model["decoder"].load_state_dict(ckpt["decoder"])

for key in model:
    model[key].eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for key in model:
    model[key].to(device)

print(f"Model on {device}")

# Round-trip test
print("\n=== Round-trip reconstruction ===")
os.makedirs("/content/reconstructed", exist_ok=True)
results = []
for fname in test_samples:
    wav, sr = torchaudio.load(fname)
    wav = wav.mean(dim=0, keepdim=True)
    wav_24k = torchaudio.functional.resample(wav, sr, 24000)
    wav_in = wav_24k.unsqueeze(0).to(device)

    # Encode
    z = model["encoder"](wav_in.float())
    _, quantized, _, _, timbre, codes = model["decoder"].quantizer(
        z, wav_in.float(), return_codes=True, n_c=2
    )
    codes_c, codes_p, codes_t, codes_r = codes
    z_c, z_p, z_t, z_r = quantized
    print(f"  {os.path.basename(fname)}: z_c={z_c.shape}, z_p={z_p.shape}, z_r={z_r.shape}")

    # Decode: upstream formula: z = z_c + z_p + z_r
    z_total = z_c.detach() + z_p.detach() + z_r.detach()
    recon = model["decoder"].inference(z_total, speaker_embedding=timbre)

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
