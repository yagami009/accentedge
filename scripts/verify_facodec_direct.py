#!/usr/bin/env python3
"""Direct FACodec reconstruction test using Amphion.

Tests: source.wav -> Amphion FACodec encode -> decode -> reconstruction.wav
No accentedge imports — tests upstream Amphion FACodec directly.

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
run("pip install -q numpy soundfile librosa scipy jiwer pyyaml huggingface-hub phonemizer speechbrain torchaudio faster-whisper pytest", "install deps")
run("test -d /content/Amphion || git clone --depth 1 https://github.com/open-mmlab/Amphion.git /content/Amphion", "clone Amphion", check=False)

os.environ["PYTHONPATH"] = "/content/Amphion:" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, "/content/Amphion")

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

print("\n=== Importing Amphion FACodec ===")
from models.codec.ns3_codec.facodec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download
print("  Amphion FACodec imported")

print("\n=== Loading Plachta/FAcodec checkpoint ===")
ckpt_path = hf_hub_download(repo_id="Plachta/FAcodec", filename="pytorch_model.bin")
config_path = hf_hub_download(repo_id="Plachta/FAcodec", filename="config.yaml")
print(f"  Checkpoint: {ckpt_path}")
print(f"  Config: {config_path}")

import yaml
from munch import Munch

with open(config_path) as f:
    config = yaml.safe_load(f)
mp = Munch(config["model_params"])

# Build encoder/decoder matching Amphion's init_facodec_models
encoder = FACodecEncoder(ngf=32, up_ratios=[2,4,5,5], out_channels=256)
decoder = FACodecDecoder(
    in_channels=256, upsample_initial_channel=1024, ngf=32,
    up_ratios=[5,5,4,2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
    vq_dim=256, codebook_dim=8,
    codebook_size_prosody=10, codebook_size_content=10, codebook_size_residual=10,
    use_gr_x_timbre=True,
)

model = {"encoder": encoder, "decoder": decoder}
ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt = ckpt.get("net", ckpt)
for key in model:
    if key in ckpt:
        model[key].load_state_dict(ckpt[key])
    model[key].eval()

print(f"  Encoder params: {sum(p.numel() for p in model['encoder'].parameters()):,}")
print(f"  Decoder params: {sum(p.numel() for p in model['decoder'].parameters()):,}")

# Move to GPU
for key in model:
    model[key] = model[key].cuda()
print("  Models on CUDA")

# Reconstruction test
print("\n=== Reconstruction Test ===")
os.makedirs("/content/reconstructed", exist_ok=True)
results = []

for fname, transcript in test_samples:
    wav, sr = torchaudio.load(fname)
    wav = wav.mean(dim=0, keepdim=True)  # mono
    wav_24k = torchaudio.functional.resample(wav, sr, 24000)

    # Encode
    z = model["encoder"](wav_24k[None, ...].cuda().float())
    _, quantized, commit_loss, codebook_loss, timbre, codes = model["decoder"].quantizer(
        z, wav_24k[None, ...].cuda().float(), return_codes=True, n_c=2
    )

    codes_c, codes_p, codes_t, codes_r = codes
    z_c, z_p, z_t, z_r = quantized

    print(f"\n  {os.path.basename(fname)}:")
    print(f"    z_c shape: {z_c.shape}")
    print(f"    z_p shape: {z_p.shape}")
    print(f"    z_r shape: {z_r.shape}")
    print(f"    timbre shape: {timbre.shape}")
    print(f"    codes_c[0] shape: {codes_c[0].shape}")
    print(f"    codes_c[1] shape: {codes_c[1].shape if codes_c[1] is not None else 'None'}")

    # Reconstruct using upstream formula: z = z_p + z_c + z_r, then decoder.inference
    recon = model["decoder"].inference(z_p.detach() + z_c.detach() + z_r.detach(), speaker_embedding=timbre)

    recon_path = f"/content/reconstructed/{os.path.basename(fname)}"
    torchaudio.save(recon_path, recon[0].cpu(), 24000)

    # SNR
    src_np = wav_24k.squeeze().cpu().numpy()
    recon_np = recon[0, 0].cpu().numpy().detach().numpy()
    min_len = min(len(src_np), len(recon_np))
    snr = 10 * np.log10(np.mean(src_np[:min_len]**2) / (np.mean((src_np[:min_len] - recon_np[:min_len])**2) + 1e-10))
    print(f"    SNR: {snr:.2f} dB")
    results.append({"file": os.path.basename(fname), "snr": float(snr)})

print("\n=== Results ===")
for r in results:
    print(f"  {r['file']}: SNR={r['snr']:.2f} dB")
passed = all(r['snr'] > 5 for r in results)
print(f"\nGATE: {'PASS' if passed else 'FAIL'} (all SNR > 5dB: {passed})")
print("\nDONE")
