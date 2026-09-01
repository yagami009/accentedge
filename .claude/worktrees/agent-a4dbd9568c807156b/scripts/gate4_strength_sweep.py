#!/usr/bin/env python3
"""
Gate 4 - Indian-English Strength Sweep
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import warnings
import soundfile as sf
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import torchaudio
import yaml
import types
from dataclasses import dataclass, field

warnings.simplefilter("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from accentedge.phase1.converter import AccentConverter
from accentedge.phase1.zc2_recomputer import ZC2Recomputer
from accentedge.phase1.denoiser import DenoisingTransformerModel
from accentedge.codec.facodec import FACodecAdapter
from accentedge.codec.interfaces import FactorizedLatents
from accentedge.evaluation.acoustic import AcousticEvaluator, mel_spectrogram
from accentedge.evaluation.identity import IdentityEvaluator
from accentedge.phase1.phoneme_pipeline import PhonemePipeline
from scripts.dataset_cmu_arctic import L2ArcticDataset

# ---------------------------------------------------------------------------
# Mock audiotools (same pattern as gate2_identity.py)
# ---------------------------------------------------------------------------

def _make_mock(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m

def install_mocks():
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

install_mocks()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(levelname)-5s | %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
logger = logging.getLogger("gate4")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class Gate4Config:
    device: str = "cuda"
    n_samples: int = 5
    output_dir: str = "artifacts/gate4"
    strengths: list = field(default_factory=lambda: [0.0, 0.25, 0.5, 0.75, 1.0])
    l2_arctic_root: str = "/data/L2-ARCTIC"
    cmu_arctic_root: str = "/data/CMU_ARCTIC"
    facodec_ckpt: str = ""
    denoiser_ckpt: str = ""
    mel_l1_max: float = 0.5
    identity_shift_min_0: float = 0.05
    identity_shift_min_1: float = 0.15
    duration_ratio_min: float = 0.9
    duration_ratio_max: float = 1.1
    snr_min: float = 10.0
    skip_wer: bool = False

# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _mel_spectrogram_l1(wav_src: np.ndarray, wav_conv: np.ndarray, sr: int = 24000) -> float:
    """Compute mel-spectrogram L1 distance between source and converted audio."""
    try:
        import librosa
        mel_src = librosa.feature.melspectrogram(
            y=wav_src, sr=sr, n_fft=1024, hop_length=256, n_mels=80
        )
        mel_conv = librosa.feature.melspectrogram(
            y=wav_conv, sr=sr, n_fft=1024, hop_length=256, n_mels=80
        )
        mel_src_db = librosa.power_to_db(mel_src, ref=np.max)
        mel_conv_db = librosa.power_to_db(mel_conv, ref=np.max)
        return float(np.mean(np.abs(mel_src_db - mel_conv_db)))
    except Exception:
        # Fallback: simple L1 on waveforms
        min_len = min(len(wav_src), len(wav_conv))
        return float(np.mean(np.abs(wav_src[:min_len] - wav_conv[:min_len])))


def _compute_snr(wav_clean: np.ndarray, wav_processed: np.ndarray) -> float:
    """Compute signal-to-noise ratio in dB."""
    min_len = min(len(wav_clean), len(wav_processed))
    wav_clean = wav_clean[:min_len]
    wav_processed = wav_processed[:min_len]
    noise = wav_clean - wav_processed
    signal_power = np.mean(wav_clean ** 2)
    noise_power = np.mean(noise ** 2)
    if noise_power < 1e-10:
        return 60.0
    return float(10 * np.log10(signal_power / noise_power))


def _compute_wer(reference: str, hypothesis: str) -> float:
    """Compute word error rate between reference and hypothesis."""
    try:
        import jiwer
        return float(jiwer.wer(reference, hypothesis))
    except ImportError:
        # Fallback: simple token-based WER
        ref_words = reference.strip().split()
        hyp_words = hypothesis.strip().split()
        if len(ref_words) == 0:
            return 0.0 if len(hyp_words) == 0 else 1.0
        # Dynamic programming for edit distance
        m, n = len(ref_words), len(hyp_words)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if ref_words[i-1] == hyp_words[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        return dp[m][n] / m

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_l2_arctic_samples(n_samples, cfg):
    """Load Indian-English (L2-ARCTIC) audio samples."""
    root = cfg.l2_arctic_root or os.environ.get("L2_ARCTIC_ROOT", "")
    if not root or not Path(root).exists():
        raise FileNotFoundError(
            f"L2-ARCTIC dataset not found at {root or '(unset)'}. "
            "Set L2_ARCTIC_ROOT env var or --l2-arctic-root."
        )

    dataset = L2ArcticDataset(root=root)
    items = dataset.get_indian_items()
    if not items:
        raise FileNotFoundError(f"No Indian-English items found in {root}")

    samples = []
    for i in range(min(n_samples, len(items))):
        wav_path, speaker, transcript = items[i]
        wav, sr = torchaudio.load(wav_path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != 24000:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=24000)
        samples.append({
            "wav": wav,
            "transcript": transcript,
            "speaker": speaker,
            "wav_path": wav_path,
        })
    return samples


def _make_synthetic_samples(n_samples):
    """Create synthetic test samples when real data is unavailable."""
    samples = []
    sr = 24000
    for i in range(n_samples):
        duration = 2.0 + i * 0.5
        t = torch.linspace(0, duration, int(sr * duration))
        freq = 100 + i * 50
        wav = torch.sin(2 * np.pi * freq * t).unsqueeze(0) * 0.3
        samples.append({
            "wav": wav,
            "transcript": f"synthetic sample {i}",
            "speaker": f"synthetic_{i}",
            "wav_path": "",
        })
    return samples

# ---------------------------------------------------------------------------
# Model building
# ---------------------------------------------------------------------------

def _build_converter(cfg, device):
    """Build AccentConverter with real FACodecAdapter and denoiser."""
    logger.info(">>> Loading FACodecAdapter...")
    facodec = FACodecAdapter(device=device.type, facodec_ckpt=cfg.facodec_ckpt)
    facodec.freeze()
    facodec.to(device)

    logger.info(">>> Loading Denoiser...")
    denoiser = DenoisingTransformerModel(
        d_model=384, nhead=6, num_layers=4, d_ff=1536,
        phone_vocab_size=393, facodec_dim=8,
    )
    if cfg.denoiser_ckpt and os.path.exists(cfg.denoiser_ckpt):
        state = torch.load(cfg.denoiser_ckpt, map_location="cpu")
        denoiser.load_state_dict(state["model"])
    denoiser.to(device)
    denoiser.eval()

    logger.info(">>> Loading PhonemePipeline...")
    phoneme_pipeline = PhonemePipeline(device=device.type)

    logger.info(">>> Building AccentConverter...")
    converter = AccentConverter(
        facodec=facodec,
        denoiser=denoiser,
        zc2_recomputer=ZC2Recomputer(),
        phoneme_pipeline=phoneme_pipeline,
        num_diffusion_steps=100,
    )
    return converter

# ---------------------------------------------------------------------------
# WER evaluator (optional, with graceful fallback)
# ---------------------------------------------------------------------------

class WEREvaluator:
    """Optional WER evaluation using faster-whisper."""

    def __init__(self, device="cuda"):
        self.available = False
        try:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("small", device=device, compute_type="float16")
            self.available = True
            logger.info(">>> faster-whisper loaded (WER evaluation enabled)")
        except ImportError:
            logger.warning("faster-whisper not available - WER will be skipped")

    @torch.no_grad()
    def transcribe(self, wav_np: np.ndarray) -> str:
        if not self.available:
            return ""
        try:
            segments, _ = self.model.transcribe(wav_np, language="en", beam_size=5)
            return " ".join(seg.text for seg in segments)
        except Exception as e:
            logger.warning(f"Whisper transcription failed: {e}")
            return ""

# ---------------------------------------------------------------------------
# Strength sweep
# ---------------------------------------------------------------------------

def run_strength_sweep(converter, samples, strengths, evaluator, wer_eval, cfg, device):
    """Run accent conversion at multiple strength levels."""
    results = []
    audio_dir = Path(cfg.output_dir) / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    key_strengths = {0.0, 0.5, 1.0}
    total = len(samples) * len(strengths)
    done = 0

    for sample_idx, sample in enumerate(samples):
        logger.info(f"Processing sample {sample_idx + 1}/{len(samples)} | Speaker: {sample['speaker']}")
        logger.info(f"Transcript: {sample['transcript'][:80]}{'...' if len(sample['transcript']) > 80 else ''}")

        wav = sample["wav"].to(device)
        transcript = sample["transcript"]
        sample_result = {
            "sample_idx": sample_idx,
            "speaker": sample["speaker"],
            "transcript": transcript,
            "wav_path": sample["wav_path"],
            "strengths": {},
        }

        for strength in strengths:
            done += 1
            logger.info(f"  [{done}/{total}] Strength {strength}: ", end="")
            t0 = time.time()

            try:
                converted_wav, metadata = converter.convert(wav, transcript, strength)
            except Exception as e:
                logger.error(f"Conversion failed: {e}")
                sample_result["strengths"][str(strength)] = {
                    "error": str(e),
                    "skipped": True,
                }
                continue

            # Move to CPU for metrics
            wav_src = wav.cpu().squeeze().numpy()
            wav_conv = converted_wav.cpu().squeeze().numpy()

            # 1. mel L1
            mel_l1_val = _mel_spectrogram_l1(wav_src, wav_conv, sr=24000)

            # 2. Identity shift
            try:
                identity_drop = evaluator.identity_drop(
                    wav_src, wav_src, wav_conv, sr=24000
                )
                identity_shift = float(identity_drop.get("identity_drop", 0.0))
            except Exception as e:
                identity_shift = None
                logger.warning(f"Identity shift computation failed: {e}")

            # 3. Duration ratio
            duration_ratio = len(wav_conv) / max(len(wav_src), 1)

            # 4. SNR
            snr_val = _compute_snr(wav_src, wav_conv)

            result = {
                "mel_l1": mel_l1_val,
                "identity_shift": identity_shift,
                "duration_ratio": duration_ratio,
                "snr_db": snr_val,
                "elapsed_sec": time.time() - t0,
            }

            # 5. WER (optional)
            if not cfg.skip_wer and wer_eval.available and identity_shift is not None:
                hyp = wer_eval.transcribe(wav_conv)
                ref_clean = transcript.upper()
                hyp_clean = hyp.upper() if hyp else ""
                wer_val = _compute_wer(ref_clean, hyp_clean)
                result["wer"] = wer_val
                result["hypothesis"] = hyp
                logger.info(f"WER={wer_val:.3f} | ", end="")

            # 6. Save audio at key strengths
            if strength in key_strengths:
                stem = Path(sample["wav_path"]).stem if sample["wav_path"] else f"sample_{sample_idx}"
                fname = f"{stem}_str{strength}.wav"
                sf.write(str(audio_dir / fname), wav_conv, 24000)

            sample_result["strengths"][str(strength)] = result
            id_str = f"{identity_shift:.4f}" if identity_shift is not None else "N/A"
            logger.info(f"mel_l1={mel_l1_val:.4f} | id_shift={id_str} | "
                        f"dur={duration_ratio:.3f} | snr={snr_val:.1f}dB | {result['elapsed_sec']:.1f}s")

        results.append(sample_result)

    return results

# ---------------------------------------------------------------------------
# Aggregation and gate evaluation
# ---------------------------------------------------------------------------

def aggregate_results(results, strengths):
    """Aggregate per-sample results into strength curves with means, stds, per-sample data."""
    curves = {}
    for s in strengths:
        mel_vals = []
        id_vals = []
        dur_vals = []
        snr_vals = []
        wer_vals = []
        for r in results:
            sr = r["strengths"].get(str(s))
            if sr is None or sr.get("skipped"):
                continue
            mel_vals.append(sr["mel_l1"])
            dur_vals.append(sr["duration_ratio"])
            snr_vals.append(sr["snr_db"])
            if sr.get("identity_shift") is not None:
                id_vals.append(sr["identity_shift"])
            if "wer" in sr:
                wer_vals.append(sr["wer"])

        curves[str(s)] = {
            "mel_l1_mean": float(np.mean(mel_vals)) if mel_vals else 0.0,
            "mel_l1_std": float(np.std(mel_vals)) if mel_vals else 0.0,
            "mel_l1_per_sample": mel_vals,
            "identity_shift_mean": float(np.mean(id_vals)) if id_vals else 0.0,
            "identity_shift_std": float(np.std(id_vals)) if id_vals else 0.0,
            "identity_shift_per_sample": id_vals,
            "duration_ratio_mean": float(np.mean(dur_vals)) if dur_vals else 0.0,
            "duration_ratio_std": float(np.std(dur_vals)) if dur_vals else 0.0,
            "duration_ratio_per_sample": dur_vals,
            "snr_db_mean": float(np.mean(snr_vals)) if snr_vals else 0.0,
            "snr_db_std": float(np.std(snr_vals)) if snr_vals else 0.0,
            "snr_db_per_sample": snr_vals,
            "wer_mean": float(np.mean(wer_vals)) if wer_vals else None,
            "wer_std": float(np.std(wer_vals)) if wer_vals else None,
            "wer_per_sample": wer_vals if wer_vals else None,
            "n_samples": len(mel_vals),
        }
    return curves


def evaluate_gate(curves, strengths, cfg):
    """Evaluate Gate 4 pass criteria."""
    failures = []
    warnings_out = []

    # Get mean identity shifts per strength
    shifts = [curves[str(s)]["identity_shift_mean"] for s in strengths]

    # 1. Identity shift at 0 ≈ 0
    if shifts[0] > cfg.identity_shift_min_0:
        failures.append(
            f"Identity shift at strength=0 is {shifts[0]:.4f} (max {cfg.identity_shift_min_0})"
        )

    # 2. Identity shift at 1 > threshold
    if shifts[-1] < cfg.identity_shift_min_1:
        failures.append(
            f"Identity shift at strength=1 is {shifts[-1]:.4f} (min {cfg.identity_shift_min_1})"
        )

    # 3. Monotonically increasing
    for i in range(1, len(shifts)):
        if shifts[i] < shifts[i - 1] * 0.9:  # 10% tolerance
            failures.append(
                f"Identity shift not monotonically increasing: "
                f"{shifts[i-1]:.4f} -> {shifts[i]:.4f}"
            )

    # 4. mel L1 below threshold
    for s in strengths:
        mel = curves[str(s)]["mel_l1_mean"]
        if mel > cfg.mel_l1_max:
            failures.append(
                f"mel L1 at strength={s} is {mel:.4f} (max {cfg.mel_l1_max})"
            )

    # 5. Duration ratio
    for s in strengths:
        dr = curves[str(s)]["duration_ratio_mean"]
        if not (cfg.duration_ratio_min <= dr <= cfg.duration_ratio_max):
            failures.append(
                f"Duration ratio at strength={s} is {dr:.3f} "
                f"(expected [{cfg.duration_ratio_min}, {cfg.duration_ratio_max}])"
            )

    # 6. SNR
    for s in strengths:
        snr = curves[str(s)]["snr_db_mean"]
        if snr < cfg.snr_min:
            warnings_out.append(
                f"SNR at strength={s} is {snr:.1f} dB (min {cfg.snr_min})"
            )

    passed = len(failures) == 0
    return passed, failures, warnings_out

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Gate 4 - Strength Sweep")
    p.add_argument("--device", default="cuda", help="torch device")
    p.add_argument("--n-samples", type=int, default=5, help="number of samples")
    p.add_argument("--output-dir", default="artifacts/gate4", help="output directory")
    p.add_argument("--strengths", default="0.0,0.25,0.5,0.75,1.0",
                    help="comma-separated strength levels")
    p.add_argument("--l2-arctic-root", default=None, help="L2-ARCTIC dataset path")
    p.add_argument("--cmu-arctic-root", default=None, help="CMU ARCTIC dataset path")
    p.add_argument("--facodec-ckpt", default=None, help="FACodec checkpoint path")
    p.add_argument("--denoiser-ckpt", default=None, help="Denoiser checkpoint path")
    p.add_argument("--checkpoint", default=None, help="combined model checkpoint (alias for facodec)")
    p.add_argument("--no-wer", action="store_true", help="skip WER evaluation")
    return p.parse_args(argv)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    args = parse_args(argv)

    cfg = Gate4Config()
    cfg.device = args.device
    cfg.n_samples = args.n_samples
    cfg.output_dir = args.output_dir
    cfg.strengths = [float(s) for s in args.strengths.split(",")]
    cfg.skip_wer = args.no_wer
    if args.l2_arctic_root:
        cfg.l2_arctic_root = args.l2_arctic_root
    if args.cmu_arctic_root:
        cfg.cmu_arctic_root = args.cmu_arctic_root
    cfg.facodec_ckpt = args.facodec_ckpt or args.checkpoint or ""
    cfg.denoiser_ckpt = args.denoiser_ckpt or ""

    device = torch.device(cfg.device)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Gate 4 - Strength Sweep")
    logger.info(f"Timestamp: {datetime.now().isoformat()}")
    logger.info(f"Strengths: {cfg.strengths}")
    logger.info(f"Samples: {cfg.n_samples}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {out_dir}")

    # Load data
    try:
        samples = _load_l2_arctic_samples(cfg.n_samples, cfg)
        logger.info(f"Loaded {len(samples)} L2-ARCTIC samples")
    except FileNotFoundError as e:
        logger.warning(f"{e}")
        logger.warning("Falling back to synthetic test samples.")
        samples = _make_synthetic_samples(cfg.n_samples)

    # Build converter
    try:
        converter = _build_converter(cfg, device)
    except Exception as e:
        logger.error(f"Cannot load converter: {e}")
        logger.error("Gate 4 requires a real FACodec checkpoint. Skipping.")
        return 1

    # Evaluators
    logger.info(">>> Loading IdentityEvaluator...")
    evaluator = IdentityEvaluator(device=device.type)
    acoustic = AcousticEvaluator(sr=24000)
    wer_eval = WEREvaluator(device=device.type)

    # Run sweep
    results = run_strength_sweep(
        converter, samples, cfg.strengths, evaluator, wer_eval, cfg, device
    )

    # Aggregate
    curves = aggregate_results(results, cfg.strengths)

    # Gate evaluation
    gate_passed, failures, warnings_out = evaluate_gate(curves, cfg.strengths, cfg)

    # Save metrics per sample
    with open(out_dir / "metrics_per_sample.json", "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved: {out_dir / 'metrics_per_sample.json'}")

    # Save strength curves
    output = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "strengths": cfg.strengths,
            "n_samples": cfg.n_samples,
            "device": cfg.device,
            "mel_l1_max": cfg.mel_l1_max,
            "identity_shift_min_0": cfg.identity_shift_min_0,
            "identity_shift_min_1": cfg.identity_shift_min_1,
            "duration_ratio_min": cfg.duration_ratio_min,
            "duration_ratio_max": cfg.duration_ratio_max,
            "snr_min": cfg.snr_min,
        },
        "strength_curves": curves,
        "gate_evaluation": {
            "passed": gate_passed,
            "failures": failures,
            "warnings": warnings_out,
        },
    }
    with open(out_dir / "strength_curves.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"Saved: {out_dir / 'strength_curves.json'}")

    # Summary
    print(f"\n{'='*60}")
    print(f"Gate 4 Results")
    print(f"{'='*60}")
    for s in cfg.strengths:
        c = curves[str(s)]
        print(f"\nStrength {s}:")
        print(f"  mel L1 mean: {c['mel_l1_mean']:.4f} +/- {c['mel_l1_std']:.4f}")
        print(f"  Identity shift: {c['identity_shift_mean']:.4f} +/- {c['identity_shift_std']:.4f}")
        print(f"  Duration ratio: {c['duration_ratio_mean']:.3f} +/- {c['duration_ratio_std']:.3f}")
        print(f"  SNR: {c['snr_db_mean']:.1f} +/- {c['snr_db_std']:.1f} dB")
        if c.get("wer_mean") is not None:
            print(f"  WER: {c['wer_mean']:.3f} +/- {c['wer_std']:.3f}")
    print(f"\nGate 4 {'PASSED' if gate_passed else 'FAILED'}")
    if failures:
        for f in failures:
            print(f"  FAIL: {f}")
    if warnings_out:
        for w in warnings_out:
            print(f"  WARN: {w}")
    print(f"\nSaved: {out_dir / 'strength_curves.json'}")
    print(f"Saved: {out_dir / 'metrics_per_sample.json'}")
    print(f"Audio: {out_dir / 'audio'}")

    return 0 if gate_passed else 1

if __name__ == "__main__":
    sys.exit(main())
