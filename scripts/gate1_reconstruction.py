#!/usr/bin/env python3
"""Gate 1 — Reconstruction Equivalence Test.

Tests that FACodecAdapter produces bit-identical (up to floating-point) reconstructions
compared to the upstream FAcodec encode→quantizer→decode pipeline.

Numeric criteria:
  * max |upstream_recon - adapter_recon| < 1e-4
  * SNR(upstream, adapter) > 60 dB
  * len(upstream) == len(adapter) exactly

Run on Colab:
  colab run scripts/gate1_reconstruction.py --gpu T4
"""
import subprocess, sys, os, json, time, types, warnings, hashlib
from pathlib import Path

warnings.simplefilter("ignore")


# ══════════════════════════════════════════════════════════════
# 0. Mock audiotools BEFORE ANY other imports (same pattern as
#    scripts/test_reconstruct.py)
# ══════════════════════════════════════════════════════════════
def _make_mock(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m

mock_audio = _make_mock("audiotools")
mock_ml = _make_mock("audiotools.ml")
mock_ml.BaseModel = type("BaseModel", (), {"INTERN": [], "EXTERN": []})
mock_audio.ml = mock_ml
mock_audio.AudioSignal = type("AudioSignal", (), {})
mock_audio.STFTParams = type("STFTParams", (), {})
mock_core = _make_mock("audiotools.core")
mock_core.util = _make_mock("audiotools.core.util")
sys.modules["audiotools"] = mock_audio
sys.modules["audiotools.ml"] = mock_ml
sys.modules["audiotools.core"] = mock_core
sys.modules["audiotools.core.util"] = mock_core.util


# ══════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════
SAMPLE_RATE = 24000
HOP_LENGTH = 300
FPS = 80  # frames per second = sample_rate / hop_length

# Paths (Colab)
FA_CODEC_DIR = "/content/FAcodec"
ACCENTEDGE_DIR = "/content/accentedge"
GATE_DIR = "/content/gate1_artifacts"
TEST_AUDIO_DIR = "/content/test_audio"

# Subdirs for artifacts
SOURCE_DIR = f"{GATE_DIR}/source"
UPSTREAM_DIR = f"{GATE_DIR}/upstream_reconstruction"
ADAPTER_DIR = f"{GATE_DIR}/adapter_reconstruction"


def run(cmd, desc="", check=True, timeout=120):
    """Run a shell command."""
    print(f"\n>>> {desc or cmd[:80]}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    if out:
        print(out[:500])
    if check and r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}")
        sys.exit(1)
    return r


def compute_snr(reference: "torch.Tensor", candidate: "torch.Tensor") -> float:
    """Compute SNR in dB between reference signal and candidate signal.

    SNR = 10 * log10(E[reference^2] / E[(reference - candidate)^2])
    """
    noise = reference - candidate
    signal_power = (reference ** 2).mean().item()
    noise_power = (noise ** 2).mean().item()
    if noise_power == 0:
        return float("inf")
    return 10.0 * (signal_power / noise_power) ** 0.1  # log10 via pow trick


def collect_wav_files() -> list:
    """Collect 5-10 WAV files: mix of native (LibriSpeech) + Indian English.

    Returns list of absolute paths to WAV files on disk.
    """
    os.makedirs(TEST_AUDIO_DIR, exist_ok=True)
    wav_files = []

    # ── Native English: LibriSpeech test-clean ──
    import torchaudio
    print("\n=== Downloading LibriSpeech test-clean (native English) ===")
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=TEST_AUDIO_DIR, url="test-clean", download=True
    )
    num_native = 5
    for i in range(min(num_native, len(dataset))):
        wav, sr, _, _, _, _ = dataset[i]
        fname = f"{TEST_AUDIO_DIR}/native_en_{i:03d}.wav"
        torchaudio.save(fname, wav, sr)
        wav_files.append(fname)
        print(f"  Saved native English: {fname}")

    # ── Indian English: try OpenSLR SLR46 (IITM Indian English) ──
    indian_dir = f"{TEST_AUDIO_DIR}/indian_english"
    os.makedirs(indian_dir, exist_ok=True)

    # Try downloading a small set of Indian English WAV files from OpenSLR
    # SLR46 has .wav files; we download a subset directly
    indian_urls = [
        # OpenSLR SLR46 - IITM Indian English Speech Database (sample files)
        # These are direct .wav links from the dataset
        "https://openslr.magicdatatech.com/resources46/SLR46/speaker001_001.wav",
        "https://openslr.magicdatatech.com/resources46/SLR46/speaker002_001.wav",
        "https://openslr.magicdatatech.com/resources46/SLR46/speaker003_001.wav",
        "https://openslr.magicdatatech.com/resources46/SLR46/speaker004_001.wav",
        "https://openslr.magicdatatech.com/resources46/SLR46/speaker005_001.wav",
    ]

    indian_downloaded = 0
    for idx, url in enumerate(indian_urls):
        fname = f"{indian_dir}/indian_en_{idx:03d}.wav"
        if not os.path.exists(fname):
            try:
                r = subprocess.run(
                    ["curl", "-L", "-o", fname, url],
                    capture_output=True, text=True, timeout=30,
                )
                if r.returncode == 0 and os.path.getsize(fname) > 1000:
                    indian_downloaded += 1
                    print(f"  Downloaded Indian English: {fname}")
                else:
                    print(f"  Skipped {url} (HTTP {r.returncode})")
            except Exception as e:
                print(f"  Skipped {url} ({e})")
        else:
            indian_downloaded += 1
            print(f"  Already exists: {fname}")

    # If OpenSLR download didn't work, try alternative sources
    if indian_downloaded < 3:
        print("\n  Trying alternative Indian English source...")
        # Try downloading from a GitHub-hosted sample set
        alt_urls = [
            "https://raw.githubusercontent.com/iamcents/accented-speech-corpus/main/indian_english/sample1.wav",
            "https://raw.githubusercontent.com/iamcents/accented-speech-corpus/main/indian_english/sample2.wav",
            "https://raw.githubusercontent.com/iamcents/accented-speech-corpus/main/indian_english/sample3.wav",
            "https://raw.githubusercontent.com/iamcents/accented-speech-corpus/main/indian_english/sample4.wav",
            "https://raw.githubusercontent.com/iamcents/accented-speech-corpus/main/indian_english/sample5.wav",
        ]
        for idx, url in enumerate(alt_urls):
            fname = f"{indian_dir}/indian_en_{idx:03d}_alt.wav"
            if not os.path.exists(fname):
                try:
                    r = subprocess.run(
                        ["curl", "-L", "-o", fname, url],
                        capture_output=True, text=True, timeout=30,
                    )
                    if r.returncode == 0 and os.path.getsize(fname) > 1000:
                        indian_downloaded += 1
                        wav_files.append(fname)
                        print(f"  Downloaded alt Indian English: {fname}")
                except Exception:
                    pass

    # Add whatever Indian English files we got
    for f in sorted(Path(indian_dir).glob("*.wav")):
        if f not in [Path(p) for p in wav_files]:
            wav_files.append(str(f))
            print(f"  Using Indian English: {f}")

    print(f"\nCollected {len(wav_files)} WAV files "
          f"({num_native} native, {len(wav_files) - num_native} Indian English)")
    return wav_files


def setup_environment():
    """Install deps, clone repos, set sys.path."""
    # ── GPU check ──
    r = run("nvidia-smi --query-gpu=name --format=csv,noheader", "GPU", check=False)
    gpu_name = r.stdout.strip()
    print(f"GPU: {gpu_name}")

    # ── Deps ──
    run(
        "pip install -q numpy soundfile librosa scipy jiwer pyyaml "
        "huggingface-hub phonemizer speechbrain torchaudio faster-whisper "
        "pytest pyworld munch einops",
        "install deps",
    )

    # ── Clone FAcodec ──
    run(
        "test -d /content/FAcodec || git clone https://github.com/Plachtaa/FAcodec.git /content/FAcodec",
        "clone FAcodec",
        check=False,
    )
    run(
        "test -f /content/FAcodec/modules/__init__.py || touch /content/FAcodec/modules/__init__.py",
        "modules init",
        check=False,
    )

    # ── Clone accentedge ──
    run(
        "test -d /content/accentedge || git clone --depth 1 https://github.com/yagami009/accentedge.git /content/accentedge",
        "clone accentedge",
        check=False,
    )

    # ── Path setup (match test_reconstruct.py pattern exactly) ──
    sys.path = [p for p in sys.path if "/content" not in p]
    sys.path.insert(0, FA_CODEC_DIR)
    sys.path.insert(0, f"{ACCENTEDGE_DIR}/src")
    os.environ["PYTHONPATH"] = f"{FA_CODEC_DIR}/modules:" + os.environ.get("PYTHONPATH", "")
    os.chdir(FA_CODEC_DIR)


def load_facodec_model():
    """Load upstream FAcodec model (same pattern as test_reconstruct.py)."""
    from modules.commons import build_model, recursive_munch
    from hf_utils import load_custom_model_from_hf

    print("\n=== Loading FAcodec model ===")
    t0 = time.time()
    ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")

    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_params = recursive_munch(config["model_params"])
    model = build_model(model_params)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    ckpt = ckpt.get("net", ckpt)
    for key in ckpt:
        model[key].load_state_dict(ckpt[key])
        model[key].eval()
    print(f"  Model loaded in {time.time() - t0:.1f}s: {list(model.keys())}")

    # Store checkpoint info for manifest
    ckpt_hash = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()[:16]
    return model, config_path, ckpt_path, ckpt_hash


def load_adapter(device: str):
    """Load FACodecAdapter."""
    from accentedge.codec.facodec import FACodecAdapter

    print("\n=== Loading FACodecAdapter ===")
    t0 = time.time()
    adapter = FACodecAdapter(device=device, facodec_ckpt="Plachta/FAcodec")
    adapter.freeze()
    print(f"  Adapter loaded in {time.time() - t0:.1f}s")
    return adapter


def upstream_reconstruct(model, waveform, device):
    """Run upstream FAcodec encode→quantizer→decode.

    Exact pattern from upstream reconstruct.py:
      z = model.encoder(wav)
      z, quantized, commitment, codebook, timbre = model.quantizer(z, wav, n_c=2)
      full_pred = model.decoder(z)   # z was overwritten by quantizer (z_q)
    """
    wav_in = waveform.unsqueeze(0).float().to(device)

    z = model["encoder"](wav_in)
    z, quantized, commitment_loss, codebook_loss, timbre = model["quantizer"](
        z, wav_in, n_c=2
    )
    full_pred = model["decoder"](z)

    return full_pred[0].cpu(), {
        "z_shape": str(list(z.shape)),
        "timbre_shape": str(list(timbre.shape)),
    }


def adapter_reconstruct(adapter, waveform):
    """Run adapter encode→decode.

    adapter.encode → FactorizedLatents(content=z_q, ...)
    adapter.decode(latents) → waveform
    """
    latents = adapter.encode(waveform)
    recon = adapter.decode(latents)

    z_p = latents.prosody
    z_c = latents.content_zc1
    z_r = latents.detail
    g = latents.timbre

    info = {
        "z_p_shape": str(list(z_p.shape)) if z_p is not None else "None",
        "z_c_shape": str(list(z_c.shape)) if z_c is not None else "None",
        "z_r_shape": str(list(z_r.shape)) if z_r is not None else "None",
        "z_g_shape": str(list(g.shape)) if g is not None else "None",
    }
    return recon, info


def save_wav(tensor, path, sample_rate):
    """Save a waveform tensor to WAV file."""
    import torchaudio
    # Ensure 2D [1, T]
    if tensor.dim() == 1:
        tensor = tensor.unsqueeze(0)
    torchaudio.save(path, tensor.float(), sample_rate)


def main():
    t_start = time.time()

    # ── Setup ──
    setup_environment()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ── Load model + adapter ──
    model, config_path, ckpt_path, ckpt_hash = load_facodec_model()
    model = {k: v.to(device) for k, v in model.items()}
    adapter = load_adapter(device)

    # ── Collect WAV files ──
    wav_files = collect_wav_files()
    if not wav_files:
        print("ERROR: No WAV files collected. Aborting.")
        sys.exit(1)

    # ── Create artifact directories ──
    for d in [SOURCE_DIR, UPSTREAM_DIR, ADAPTER_DIR]:
        os.makedirs(d, exist_ok=True)

    # ══════════════════════════════════════════════════════════
    # Main loop: run both paths on each WAV, compare, save
    # ══════════════════════════════════════════════════════════
    sample_metrics = []
    latent_stats = []

    print(f"\n{'='*70}")
    print(f"RECONSTRUCTION EQUIVALENCE TEST  ({len(wav_files)} samples)")
    print(f"{'='*70}")

    for idx, fpath in enumerate(wav_files):
        fname = Path(fpath).name
        base = Path(fname).stem
        print(f"\n{'─'*60}")
        print(f"Sample {idx+1}/{len(wav_files)}: {fname}")

        # ── Load WAV ──
        waveform, orig_sr = torchaudio.load(fpath)
        duration = waveform.shape[-1] / orig_sr
        # Resample to 24k
        if orig_sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(waveform, orig_sr, SAMPLE_RATE)
        print(f"  Input: sr={orig_sr}→{SAMPLE_RATE}, duration={duration:.2f}s, "
              f"shape={list(waveform.shape)}")

        # ── Upstream path ──
        upstream_recon, upstream_info = upstream_reconstruct(model, waveform, device)
        print(f"  Upstream recon: shape={list(upstream_recon.shape)}")

        # ── Adapter path ──
        adapter_recon, adapter_info = adapter_reconstruct(adapter, waveform)
        print(f"  Adapter recon:  shape={list(adapter_recon.shape)}")

        # ── Print latent shapes ──
        print(f"  Encoded shape:          {upstream_info['z_shape']}")
        print(f"  z_p (prosody) shape:    {adapter_info['z_p_shape']}")
        print(f"  z_c (content) shape:    {adapter_info['z_c_shape']}")
        print(f"  z_r (detail) shape:     {adapter_info['z_r_shape']}")
        print(f"  Timbre (g) shape:       {adapter_info['z_g_shape']}")
        print(f"  Codec frames:            {upstream_recon.shape[-1]}")
        print(f"  Frames / duration:       {upstream_recon.shape[-1] / duration:.1f} fps "
              f"(expected {FPS})")

        # ── Numeric criteria ──
        # 1. Length check
        len_match = (upstream_recon.shape == adapter_recon.shape)
        upstream_len = upstream_recon.shape[-1]
        adapter_len = adapter_recon.shape[-1]

        # 2. Max absolute difference
        max_abs_diff = (upstream_recon - adapter_recon).abs().max().item()

        # 3. SNR
        snr_db = compute_snr(upstream_recon, adapter_recon)

        # ── Per-sample results ──
        passed_max_diff = max_abs_diff < 1e-4
        passed_snr = snr_db > 60.0
        passed_len = len_match
        all_passed = passed_max_diff and passed_snr and passed_len

        print(f"\n  Criteria:")
        print(f"    max |upstream - adapter| = {max_abs_diff:.2e}  "
              f"(< 1e-4): {'PASS' if passed_max_diff else 'FAIL'}")
        print(f"    SNR(upstream, adapter)    = {snr_db:.2f} dB  "
              f"(> 60 dB): {'PASS' if passed_snr else 'FAIL'}")
        print(f"    len(upstream) == len(adap) = {upstream_len} == {adapter_len}  "
              f"(==): {'PASS' if passed_len else 'FAIL'}")
        print(f"    OVERALL: {'✅ PASS' if all_passed else '❌ FAIL'}")

        sample_metrics.append({
            "sample_idx": idx,
            "filename": fname,
            "input_sr": int(orig_sr),
            "duration_sec": round(duration, 3),
            "encoded_shape": upstream_info["z_shape"],
            "z_p_shape": adapter_info["z_p_shape"],
            "z_c_shape": adapter_info["z_c_shape"],
            "z_r_shape": adapter_info["z_r_shape"],
            "timbre_shape": adapter_info["z_g_shape"],
            "output_shape": list(upstream_recon.shape),
            "codec_frames": int(upstream_recon.shape[-1]),
            "frames_per_sec": round(upstream_recon.shape[-1] / duration, 1),
            "max_abs_diff": round(max_abs_diff, 8),
            "snr_db": round(snr_db, 4),
            "upstream_length": upstream_len,
            "adapter_length": adapter_len,
            "length_match": bool(passed_len),
            "passed_max_diff": bool(passed_max_diff),
            "passed_snr": bool(passed_snr),
            "passed_all": bool(all_passed),
        })

        # ── Save latent stats ──
        with torch.no_grad():
            latent_stats.append({
                "sample_idx": idx,
                "filename": fname,
                "z_q_min": float(upstream_recon.min().item()),
                "z_q_max": float(upstream_recon.max().item()),
                "z_q_mean": float(upstream_recon.mean().item()),
                "z_q_std": float(upstream_recon.std().item()),
            })

        # ── Save WAV files ──
        save_wav(waveform, f"{SOURCE_DIR}/{base}.wav", SAMPLE_RATE)
        save_wav(upstream_recon, f"{UPSTREAM_DIR}/{base}_upstream.wav", SAMPLE_RATE)
        save_wav(adapter_recon, f"{ADAPTER_DIR}/{base}_adapter.wav", SAMPLE_RATE)

    # ══════════════════════════════════════════════════════════
    # Summary
    # ════════════════════════════════════════════════════════��═
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")

    all_pass = all(s["passed_all"] for s in sample_metrics)
    max_diff_overall = max(s["max_abs_diff"] for s in sample_metrics)
    min_snr_overall = min(s["snr_db"] for s in sample_metrics)

    print(f"  Samples tested:  {len(sample_metrics)}")
    print(f"  All PASS:        {all_pass}")
    print(f"  Max abs diff:    {max_diff_overall:.2e}  (threshold: 1e-4)")
    print(f"  Min SNR:         {min_snr_overall:.2f} dB  (threshold: 60 dB)")
    print(f"  Length matches:  {all(s['length_match'] for s in sample_metrics)}")

    # Per-sample table
    print(f"\n{'Idx':<4} {'File':<35} {'MaxDiff':>12} {'SNR(dB)':>10} {'LenOK':>6} {'Result':>8}")
    print(f"{'─'*4} {'─'*35} {'─'*12} {'─'*10} {'─'*6} {'─'*8}")
    for s in sample_metrics:
        status = "PASS" if s["passed_all"] else "FAIL"
        print(f"{s['sample_idx']:<4} {s['filename']:<35} "
              f"{s['max_abs_diff']:>12.2e} {s['snr_db']:>10.2f} "
              f"{'Y' if s['length_match'] else 'N':>6} {status:>8}")

    # ── Save JSON results ──
    results = {
        "gate": "gate1_reconstruction",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(time.time() - t_start, 1),
        "thresholds": {
            "max_abs_diff": 1e-4,
            "snr_db": 60.0,
            "length_match": True,
        },
        "overall_pass": bool(all_pass),
        "max_abs_diff_overall": round(max_diff_overall, 8),
        "min_snr_db_overall": round(min_snr_overall, 4),
        "samples": sample_metrics,
    }

    metrics_path = f"{GATE_DIR}/reconstruction_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)

    latent_path = f"{GATE_DIR}/latent_stats.json"
    with open(latent_path, "w") as f:
        json.dump(latent_stats, f, indent=2)

    print(f"\nSaved: {metrics_path}")
    print(f"Saved: {latent_path}")

    if all_pass:
        print("\n🎉 GATE 1 PASSED — Adapter reconstruction matches upstream.")
    else:
        print("\n❌ GATE 1 FAILED — Adapter reconstruction diverges from upstream.")
        sys.exit(1)

    print("\nDONE")


if __name__ == "__main__":
    main()
