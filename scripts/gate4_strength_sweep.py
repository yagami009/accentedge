#!/usr/bin/env python3
"""Gate 4 — Indian-English Strength Sweep

Runs accent conversion at multiple strength levels on Indian-English utterances,
measuring identity shift, acoustic quality, and content preservation.

Usage:
    python scripts/gate4_strength_sweep.py \
        --device cuda \
        --n-samples 5 \
        --output-dir artifacts/gate4 \
        --strengths 0.0,0.25,0.5,0.75,1.0

Environment:
    FACODEC_CKPT  — FAcodec checkpoint name (default: Plachta/FAcodec)
    DENOISER_CKPT — denoiser checkpoint path (optional)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import types
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import librosa

warnings.simplefilter("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

class Gate4Config:
    output_dir: Path = Path("artifacts/gate4")
    device: str = "cpu"
    n_samples: int = 5
    strengths: list[float] = [0.0, 0.25, 0.5, 0.75, 1.0]
    facodec_ckpt: str = os.environ.get("FACODEC_CKPT", "Plachta/FAcodec")
    denoiser_ckpt: str = os.environ.get("DENOISER_CKPT", "")
    target_sr: int = 24000
    l1_filter: list[str] = ["Hindi"]
    seed: int = 42

    # Gate 4 pass thresholds
    max_mel_l1: float = 0.5
    max_identity_at_0: float = 0.02
    min_identity_at_1: float = 0.15


cfg = Gate4Config()


# ═══════════════════════════════════════════════════════════════════════════════
# Mock audiotools
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock(name: str):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m


def _install_mocks():
    """Install mock audiotools modules before importing FAcodec."""
    mock_audio = _make_mock("audiotools")
    mock_ml = _make_mock("audiotools.ml")
    sys.modules["audiotools"] = mock_audio
    sys.modules["audiotools.ml"] = mock_ml
    mock_ml.BaseModel = type("BaseModel", (), {"__init__": lambda *a, **k: None})


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def load_indian_samples(n_samples: int = 5) -> list:
    """Load L2-ARCTIC Indian English utterances."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from accentedge.scripts.dataset_cmu_arctic import L2ArcticDataset

    print(f"\nLoading L2-ARCTIC dataset (filtering for L1={cfg.l1_filter})...")
    ds = L2ArcticDataset(root=None)

    # Get items filtered by L1 language
    all_items = ds.all_items()
    filtered = [(p, spk, t) for p, spk, t in all_items if spk in cfg.l1_filter]

    if not filtered:
        print("  WARNING: No exact L1 matches, using first available items.")
        filtered = all_items[:n_samples]

    import random
    rng = random.Random(cfg.seed)
    rng.shuffle(filtered)

    samples = []
    for path, speaker, transcript in filtered[:n_samples]:
        wav, sr = torchaudio.load(path)
        if sr != cfg.target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=cfg.target_sr)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        wav = wav.to(torch.float32)
        samples.append({
            "id": Path(path).stem,
            "path": path,
            "speaker": speaker,
            "transcript": transcript,
            "wav": wav,
            "duration": float(wav.shape[-1]) / cfg.target_sr,
        })

    print(f"  Loaded {len(samples)} Indian English utterances")
    for s in samples:
        print(f"    {s['id']} | {s['speaker']} | {s['duration']:.2f}s")
    return samples


# ═══════════════════════════════════════════════════════════════════════════════
# FAcodec + AccentConverter
# ═══════════════════════════════════════════════════════════════════════════════

def load_facodec_and_converter(device: str) -> tuple | None:
    """Load FACodecAdapter and AccentConverter. Returns None if unavailable."""
    _install_mocks()

    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from accentedge.codec.facodec import FACodecAdapter
        from accentedge.phase1.converter import AccentConverter
    except ImportError as e:
        print(f"  FAIL: Could not import modules: {e}")
        return None

    print(f"\nLoading FACodec checkpoint: {cfg.facodec_ckpt}")
    try:
        codec = FACodecAdapter.from_pretrained(cfg.facodec_ckpt, device=device)
        codec.freeze()
        print(f"  FACodec loaded successfully")
    except Exception as e:
        print(f"  FAIL: Could not load FACodec: {e}")
        return None

    print(f"Initializing AccentConverter...")
    try:
        converter = AccentConverter(codec=codec, device=device)

        # Load denoiser checkpoint if provided
        if cfg.denoiser_ckpt and Path(cfg.denoiser_ckpt).exists():
            print(f"  Loading denoiser from {cfg.denoiser_ckpt}")
            ckpt = torch.load(cfg.denoiser_ckpt, map_location=device, weights_only=True)
            converter.denoiser.load_state_dict(ckpt)
            print(f"  Denoiser loaded successfully")
        else:
            print(f"  WARNING: No denoiser checkpoint. Conversion will use zero-initialized denoiser.")
            print(f"  Set DENOISER_CKPT env var to load trained weights.")

        return codec, converter
    except Exception as e:
        print(f"  FAIL: Could not initialize converter: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation: Mel L1
# ═══════════════════════════════════════════════════════════════════════════════

def compute_mel_l1(source: np.ndarray, target: np.ndarray, sr: int = 24000) -> float:
    """Mean absolute difference between mel spectrograms."""
    mel_src = librosa.feature.melspectrogram(
        y=source, sr=sr, n_fft=2048, hop_length=300, n_mels=80
    )
    mel_tgt = librosa.feature.melspectrogram(
        y=target, sr=sr, n_fft=2048, hop_length=300, n_mels=80
    )
    mel_src_db = librosa.power_to_db(mel_src, ref=np.max)
    mel_tgt_db = librosa.power_to_db(mel_tgt, ref=np.max)

    min_len = min(mel_src_db.shape[1], mel_tgt_db.shape[1])
    diff = np.abs(mel_src_db[:, :min_len] - mel_tgt_db[:, :min_len])
    return float(diff.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation: Identity (ECAPA-TDNN)
# ═══════════════════════════════════════════════════════════════════════════════

class IdentityEvaluator:
    """ECAPA-TDNN speaker similarity evaluator."""

    def __init__(self, device: str = "cpu"):
        print("Loading SpeechBrain ECAPA-TDNN...")
        from speechbrain.pretrained import EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/spkrec_ecapa",
        )
        self.device = device
        print("  ECAPA-TDNN loaded")

    @torch.no_grad()
    def embed(self, waveform: np.ndarray, sr: int = 24000) -> torch.Tensor:
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        wav_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        emb = self.classifier.encode_batch(wav_t.to(self.device))
        return (emb / emb.norm(dim=-1, keepdim=True)).squeeze()

    def similarity(self, src: np.ndarray, tgt: np.ndarray, sr: int = 24000) -> float:
        emb_src = self.embed(src, sr)
        emb_tgt = self.embed(tgt, sr)
        return float(torch.dot(emb_src, emb_tgt).item())


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation: WER (faster-whisper)
# ════════════════════════���══════════════════════════════════════════════════════

class WEREvaluator:
    """Content preservation via faster-whisper transcription."""

    def __init__(self, device: str = "cpu"):
        self.available = False
        self.model = None
        try:
            from faster_whisper import WhisperModel
            compute_type = "float16" if device == "cuda" else "int8"
            self.model = WhisperModel("base", device=device, compute_type=compute_type)
            self.available = True
            print("  faster-whisper loaded")
        except ImportError:
            print("  WARNING: faster-whisper not available, WER will be skipped")

    def transcribe(self, waveform: np.ndarray, sr: int = 24000) -> str:
        if not self.available:
            return ""
        segments, _ = self.model.transcribe(waveform, beam_size=5, language="en")
        return " ".join(seg.text for seg in segments).strip()

    def compute_wer(self, reference: str, hypothesis: str) -> float | None:
        if not self.available or not reference or not hypothesis:
            return None
        try:
            import jiwer
            return jiwer.wer(reference, hypothesis)
        except ImportError:
            # Fallback: simple word-level accuracy
            ref_words = reference.lower().split()
            hyp_words = hypothesis.lower().split()
            if not ref_words:
                return 0.0
            # Very rough: count matching words
            matches = sum(1 for w in hyp_words if w in ref_words)
            return 1.0 - matches / max(len(ref_words), 1)


# ═══════════════════════════════════════════════════════════════════════════════
# Strength sweep runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_strength_sweep(
    samples: list,
    converter,
    strengths: list[float],
    identity_eval: IdentityEvaluator,
    wer_eval: WEREvaluator | None,
) -> tuple[list[dict], dict]:
    """Run conversion at each strength for each sample, collect metrics."""
    results = []

    for sample in samples:
        wav = sample["wav"]
        transcript = sample["transcript"]
        wav_np = wav.squeeze().cpu().numpy().astype(np.float32)

        # Pre-compute reference metrics
        recon = converter.codec.reconstruction(wav)
        recon_np = recon.squeeze().cpu().numpy().astype(np.float32)
        sim_recon = identity_eval.similarity(wav_np, recon_np)

        ref_wer = wer_eval.transcribe(wav_np) if wer_eval and wer_eval.available else None

        sample_results = {
            "id": sample["id"],
            "speaker": sample["speaker"],
            "transcript": transcript,
            "duration": sample["duration"],
            "sim_reconstruction": sim_recon,
            "strengths": {},
        }

        for strength in strengths:
            print(f"  [{sample['id']}] strength={strength:.2f}...", end=" ", flush=True)
            t0 = time.time()

            try:
                out_wav = converter.convert(wav, transcript, strength=strength)
            except Exception as e:
                print(f"FAIL ({e})")
                sample_results["strengths"][str(strength)] = {"error": str(e)}
                continue

            out_np = out_wav.squeeze().cpu().numpy().astype(np.float32)

            # Mel L1
            mel_l1 = compute_mel_l1(wav_np, out_np)

            # Identity shift
            sim_conv = identity_eval.similarity(wav_np, out_np)
            identity_shift = sim_recon - sim_conv

            # WER
            hyp_wer = wer_eval.transcribe(out_np) if wer_eval and wer_eval.available else None
            wer = wer_eval.compute_wer(ref_wer, hyp_wer) if ref_wer and hyp_wer else None

            elapsed = time.time() - t0
            print(f"mel_l1={mel_l1:.4f} id_shift={identity_shift:.4f} ({elapsed:.1f}s)")

            sample_results["strengths"][str(strength)] = {
                "mel_l1": round(mel_l1, 6),
                "sim_converted": round(sim_conv, 6),
                "identity_shift": round(identity_shift, 6),
                "wer": round(wer, 6) if wer is not None else None,
            }

        results.append(sample_results)

    return results, {}


# ═══════════════════════════════════════════════════════════════════════════════
# Curve generation
# ════════════════════════════════════════════���══════════════════════════════════

def generate_curves(results: list[dict], strengths: list[float]) -> dict:
    """Aggregate per-sample results into mean ± std curves."""
    curves = {"strengths": strengths}

    for metric in ["mel_l1", "identity_shift", "wer"]:
        means = []
        stds = []
        ns = []

        for s in strengths:
            vals = []
            for r in results:
                entry = r["strengths"].get(str(s), {})
                val = entry.get(metric)
                if val is not None and "error" not in entry:
                    vals.append(val)

            if vals:
                means.append(float(np.mean(vals)))
                stds.append(float(np.std(vals)))
                ns.append(len(vals))
            else:
                means.append(None)
                stds.append(None)
                ns.append(0)

        curves[metric] = {"mean": means, "std": stds, "n": ns}

    return curves


# ═══════════════════════════════════════════════════════════════════════��═══════
# Gate 4 verdict
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_gate4(curves: dict, cfg: Gate4Config) -> dict:
    """Check Gate 4 pass criteria."""
    strengths = curves["strengths"]
    id_means = curves["identity_shift"]["mean"]
    mel_means = curves["mel_l1"]["mean"]

    # Identity at s=0
    id_at_0 = id_means[0] if id_means[0] is not None else None
    pass_id_0 = id_at_0 is not None and id_at_0 < cfg.max_identity_at_0

    # Identity at s=1
    id_at_1 = id_means[-1] if id_means[-1] is not None else None
    pass_id_1 = id_at_1 is not None and id_at_1 > cfg.min_identity_at_1

    # Monotonic increase
    valid_means = [m for m in id_means if m is not None]
    monotonic = all(valid_means[i] <= valid_means[i + 1] for i in range(len(valid_means) - 1))

    # Mel L1 at all strengths
    mel_valid = [m for m in mel_means if m is not None]
    pass_mel = all(m < cfg.max_mel_l1 for m in mel_valid)

    overall = pass_id_0 and pass_id_1 and monotonic and pass_mel

    interpretation_parts = []
    if not pass_id_0:
        interpretation_parts.append(
            f"Identity shift at s=0 is {id_at_0:.4f}, exceeds threshold {cfg.max_identity_at_0}"
        )
    if not pass_id_1:
        interpretation_parts.append(
            f"Identity shift at s=1 is {id_at_1:.4f}, below threshold {cfg.min_identity_at_1}"
        )
    if not monotonic:
        interpretation_parts.append("Identity shift is not monotonically increasing")
    if not pass_mel:
        interpretation_parts.append(f"Mel L1 exceeds threshold at some strength")

    interpretation = "; ".join(interpretation_parts) if interpretation_parts else "All criteria met."

    return {
        "identity_at_0_pass": pass_id_0,
        "identity_at_1_pass": pass_id_1,
        "monotonic_pass": monotonic,
        "mel_l1_pass": pass_mel,
        "gate4_pass": overall,
        "identity_at_0_value": id_at_0,
        "identity_at_1_value": id_at_1,
        "mel_l1_max": max(mel_valid) if mel_valid else None,
        "interpretation": interpretation,
    }


# ═════════════════════════════════════════════════════════════════════════════���═
# Plotting
# ═══════════════════════════════════════════════════════════════════════════════

def plot_curves(curves: dict, output_path: str):
    """Generate strength curves plot."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping plot")
        return

    strengths = curves["strengths"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Gate 4 - Strength Sweep Curves", fontsize=14, fontweight="bold")

    configs = {
        "identity_shift": ("#e74c3c", "Identity Shift (1 - cosine sim)"),
        "mel_l1": ("#3498db", "Mel L1 (dB)"),
        "wer": ("#2ecc71", "Word Error Rate"),
    }

    for idx, (key, (color, label)) in enumerate(configs.items()):
        ax = axes[idx]
        means = curves[key]["mean"]
        stds = curves[key]["std"]

        if means[0] is None:
            ax.set_title(f"{label}\n(no data)")
            ax.set_xlabel("Strength")
            continue

        ps = [s for s, m in zip(strengths, means) if m is not None]
        pm = [m for m in means if m is not None]
        pstd = [st for st, m in zip(stds, means) if m is not None]

        ax.plot(ps, pm, "o-", color=color, linewidth=2, markersize=8, label="Mean")
        ax.fill_between(
            ps,
            [m - st for m, st in zip(pm, pstd)],
            [m + st for m, st in zip(pm, pstd)],
            alpha=0.2, color=color, label="±1 std",
        )

        ns = curves[key]["n"]
        for s, m, n in zip(strengths, means, ns):
            if m is not None and n > 0:
                ax.annotate(f"n={n}", (s, m), textcoords="offset points",
                           xytext=(0, 10), ha="center", fontsize=8)

        ax.set_xlabel("Conversion Strength")
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.set_xticks(strengths)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved: {output_path}")


# ═══════════════════════════════════════════════════���═══════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Gate 4 Strength Sweep")
    parser.add_argument("--device", default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--n-samples", type=int, default=5, help="Number of test samples")
    parser.add_argument("--output-dir", default="artifacts/gate4", help="Output directory")
    parser.add_argument("--strengths", default="0.0,0.25,0.5,0.75,1.0",
                        help="Comma-separated strength levels")
    args = parser.parse_args()

    cfg.device = args.device
    cfg.n_samples = args.n_samples
    cfg.output_dir = Path(args.output_dir)
    cfg.strengths = [float(s.strip()) for s in args.strengths.split(",")]

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    output_dir = cfg.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("GATE 4 - Indian-English Strength Sweep")
    print("=" * 60)
    print(f"Device: {cfg.device}")
    print(f"Strengths: {cfg.strengths}")
    print(f"Output: {output_dir}")

    start_time = time.time()

    # ── Load data ─────────────────────────────────────────────────────────────
    samples = load_indian_samples(cfg.n_samples)
    if not samples:
        print("ERROR: No samples loaded. Cannot run Gate 4.")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    model_result = load_facodec_and_converter(cfg.device)
    if model_result is None:
        print("\n" + "=" * 60)
        print("FACODEC NOT AVAILABLE - Gate 4 Skipped")
        print("=" * 60)
        print("Set FACODEC_CKPT environment variable to point to a valid")
        print("FAcodec checkpoint, e.g.: export FACODEC_CKPT=Plachta/FAcodec")
        print("=" * 60)

        # Write skip manifest
        manifest = {
            "timestamp": datetime.utcnow().isoformat(),
            "status": "skipped",
            "reason": "FACodec not available",
            "config": {
                "device": cfg.device,
                "n_samples": cfg.n_samples,
                "strengths": cfg.strengths,
                "facodec_ckpt": cfg.facodec_ckpt,
                "denoiser_ckpt": cfg.denoiser_ckpt,
            },
            "gate4_pass": False,
        }
        with open(output_dir / "gate4_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nManifest saved: {output_dir}/gate4_manifest.json")
        return

    codec, converter = model_result

    # ── Load evaluators ────────────────────────────────────────────────────────
    print("\nLoading evaluation models...")
    identity_eval = IdentityEvaluator(device=cfg.device)
    wer_eval = WEREvaluator(device=cfg.device)

    # ── Run sweep ──────────────────────────────────────────────────────────────
    print(f"\nRunning strength sweep on {len(samples)} samples...")
    results, _ = run_strength_sweep(
        samples, converter, cfg.strengths, identity_eval, wer_eval
    )

    elapsed = time.time() - start_time
    print(f"\nSweep completed in {elapsed:.1f}s")

    # ── Generate curves ────────────────────────────────────────────────────────
    print("\nGenerating curves...")
    curves = generate_curves(results, cfg.strengths)

    # ── Gate 4 verdict ─────────────────────────────────────────────────────────
    verdict = evaluate_gate4(curves, cfg)

    print("\n" + "=" * 60)
    print("GATE 4 VERDICT")
    print("=" * 60)
    print(f"  Identity @ s=0: {verdict['identity_at_0_value']:.4f} "
          f"(threshold: < {cfg.max_identity_at_0}) {'PASS' if verdict['identity_at_0_pass'] else 'FAIL'}")
    print(f"  Identity @ s=1: {verdict['identity_at_1_value']:.4f} "
          f"(threshold: > {cfg.min_identity_at_1}) {'PASS' if verdict['identity_at_1_pass'] else 'FAIL'}")
    print(f"  Monotonic:      {verdict['monotonic_pass']} "
          f"({'YES' if verdict['monotonic_pass'] else 'NO'})")
    print(f"  Mel L1 max:     {verdict['mel_l1_max']:.4f} "
          f"(threshold: < {cfg.max_mel_l1}) {'PASS' if verdict['mel_l1_pass'] else 'FAIL'}")
    print(f"  OVERALL:        {'PASSED' if verdict['gate4_pass'] else 'FAILED'}")
    print(f"  {verdict['interpretation']}")
    print("=" * 60)

    # ── Save outputs ───────────────────────────────────────────────────────────
    # Metrics per sample
    with open(output_dir / "metrics_per_sample.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_dir}/metrics_per_sample.json")

    # Curves
    with open(output_dir / "strength_curves.json", "w") as f:
        json.dump(curves, f, indent=2)
    print(f"Saved: {output_dir}/strength_curves.json")

    # Manifest
    manifest = {
        "timestamp": datetime.utcnow().isoformat(),
        "status": "completed",
        "config": {
            "device": cfg.device,
            "n_samples": cfg.n_samples,
            "strengths": cfg.strengths,
            "facodec_ckpt": cfg.facodec_ckpt,
            "denoiser_ckpt": cfg.denoiser_ckpt,
            "l1_filter": cfg.l1_filter,
            "seed": cfg.seed,
        },
        "samples": [
            {"id": s["id"], "speaker": s["speaker"], "duration": s["duration"]}
            for s in samples
        ],
        "gate4_pass": verdict["gate4_pass"],
        "criteria": verdict,
    }
    with open(output_dir / "gate4_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print(f"Saved: {output_dir}/gate4_manifest.json")

    # Plot
    plot_path = str(output_dir / "strength_curves.png")
    plot_curves(curves, plot_path)

    print(f"\nAll artifacts in: {output_dir}/")
    print(f"Gate 4: {'PASSED' if verdict['gate4_pass'] else 'FAILED'}")


if __name__ == "__main__":
    main()
