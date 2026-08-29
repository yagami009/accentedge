#!/usr/bin/env python3
"""
Parameter sweep for Seed-VC accent conversion quality optimization.

Runs the converter across a grid of (diffusion_steps, inference_cfg_rate,
length_adjust) combinations, evaluates each output, and ranks results by
composite quality score.

Usage:
    python scripts/run_parameter_sweep.py [--max-combos 20]

Results:
    JSON  → results/sweep/sweep_<timestamp>.json
    Audio → results/sweep/audio/<label>.wav
"""

import argparse
import gc
import json
import logging
import math
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.conversion.seedvc import SeedVCConverter
from src.evaluation.metrics import AudioQualityMetrics
from scripts.compute_composite_score import compute_composite_score

# ── Logging ────────────────────────────────────────��───────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Parameter grid ─────────────────────────────────────────────────────
DIFFUSION_STEPS   = [5, 10, 15, 20]
INFERENCE_CFG_RATE = [0.5, 0.7, 0.9]
LENGTH_ADJUST     = [0.8, 1.0, 1.2]

# ── File paths ─────────────────────────────────────────────────────────
SOURCE_PATH      = PROJECT_ROOT / "samples" / "source" / "indian_english_01.wav"
REFERENCE_PATH   = PROJECT_ROOT / "samples" / "reference" / "us_english_reference_01.wav"
SWEEP_DIR        = PROJECT_ROOT / "results" / "sweep"
AUDIO_DIR        = SWEEP_DIR / "audio"


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ram_mb() -> float:
    return psutil.Process().memory_info().rss / 1024 / 1024


def _load_wav(path: Path) -> tuple[np.ndarray, int]:
    """Load a WAV file as mono float32, return (audio, sample_rate)."""
    audio, sr = sf.read(str(path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32), int(sr)


def _build_grid(max_combos: int) -> list[dict]:
    """Build parameter combinations, optionally truncated."""
    combos = []
    for ds in DIFFUSION_STEPS:
        for cfg in INFERENCE_CFG_RATE:
            for la in LENGTH_ADJUST:
                combos.append({
                    "diffusion_steps": ds,
                    "inference_cfg_rate": cfg,
                    "length_adjust": la,
                })
    # Shuffle for fair early termination, then truncate
    rng = np.random.RandomState(42)
    rng.shuffle(combos)
    if max_combos is not None and max_combos < len(combos):
        combos = combos[:max_combos]
    return combos


def _combo_label(combo: dict) -> str:
    return (
        f"ds{combo['diffusion_steps']}_"
        f"cfg{combo['inference_cfg_rate']}_"
        f"la{combo['length_adjust']}"
    )


def run_sweep(max_combos: int | None = None):
    """Execute the full parameter sweep."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    sweep_ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sweep_file = SWEEP_DIR / f"sweep_{sweep_ts}.json"

    # ── Load source and reference audio ────────────────────────────────
    logger.info("Loading source audio: %s", SOURCE_PATH)
    source_audio, source_sr = _load_wav(SOURCE_PATH)
    logger.info("  Duration: %.2fs, SR: %dHz", len(source_audio) / source_sr, source_sr)

    logger.info("Loading reference audio: %s", REFERENCE_PATH)
    ref_audio, ref_sr = _load_wav(REFERENCE_PATH)
    logger.info("  Duration: %.2fs, SR: %dHz", len(ref_audio) / ref_sr, ref_sr)

    # ── Build parameter grid ───────────────────────────────────────────
    combos = _build_grid(max_combos)
    total = len(combos)
    logger.info("Parameter grid: %d × %d × %d = %d combinations",
                len(DIFFUSION_STEPS), len(INFERENCE_CFG_RATE),
                len(LENGTH_ADJUST), len(DIFFUSION_STEPS) * len(INFERENCE_CFG_RATE) * len(LENGTH_ADJUST))
    if max_combos is not None and total < len(DIFFUSION_STEPS) * len(INFERENCE_CFG_RATE) * len(LENGTH_ADJUST):
        logger.info("Truncated to %d combinations (--max-combos)", total)

    # ── Initialize converter (model loads lazily on first convert) ─────
    logger.info("Initializing SeedVCConverter (device=cpu)...")
    t_init = time.perf_counter()
    converter = SeedVCConverter(device="cpu", model_config="seed-uvit-tat-xlsr-tiny")
    logger.info("  Init: %.2fs, RAM: %.1f MB", time.perf_counter() - t_init, _ram_mb())

    # ── Initialize metrics engine ──────────────────────────────────────
    metrics_engine = AudioQualityMetrics(sample_rate=converter.sample_rate)

    # ── Run sweep ──────────────────────────────────────────────────────
    results = []
    t_sweep_start = time.perf_counter()

    for idx, combo in enumerate(combos, 1):
        label = _combo_label(combo)
        logger.info("─" * 60)
        logger.info("[%d/%d] Testing: %s", idx, total, label)
        logger.info("  diffusion_steps=%d, inference_cfg_rate=%.2f, length_adjust=%.1f",
                    combo["diffusion_steps"], combo["inference_cfg_rate"], combo["length_adjust"])

        t_combo = time.perf_counter()
        try:
            # Run conversion
            out_path = AUDIO_DIR / f"{label}.wav"
            conv_result = converter.convert(
                source_audio=source_audio,
                reference_audio=ref_audio,
                output_path=str(out_path),
                diffusion_steps=combo["diffusion_steps"],
                inference_cfg_rate=combo["inference_cfg_rate"],
                length_adjust=combo["length_adjust"],
            )

            latency_ms = conv_result.get("latency_ms", (time.perf_counter() - t_combo) * 1000)
            rtf = conv_result.get("rtf", None)

            # Load output for evaluation
            output_audio, output_sr = _load_wav(out_path)

            # Evaluate
            metric_results = metrics_engine.compute_all(source_audio, output_audio, ref_audio)

            # Convert MetricResult objects to dicts
            metrics_dict = {}
            for k, v in metric_results.items():
                if hasattr(v, "to_dict"):
                    metrics_dict[k] = v.to_dict()
                else:
                    metrics_dict[k] = {"value": float(v)}

            # Compute composite score
            composite = compute_composite_score(metrics_dict)

            entry = {
                "combo_label": label,
                "parameters": combo,
                "metrics": metrics_dict,
                "composite_score": composite,
                "latency_ms": round(latency_ms, 1),
                "rtf": round(rtf, 4) if rtf is not None else None,
                "output_path": str(out_path),
                "output_duration_seconds": round(len(output_audio) / output_sr, 3),
                "status": "success",
                "error": None,
            }
            results.append(entry)
            logger.info("  ✔ Score: %.2f, Latency: %.1f ms", composite, latency_ms)

        except Exception as e:
            entry = {
                "combo_label": label,
                "parameters": combo,
                "metrics": {},
                "composite_score": None,
                "latency_ms": round((time.perf_counter() - t_combo) * 1000, 1),
                "rtf": None,
                "output_path": None,
                "status": "failed",
                "error": str(e),
                "error_traceback": traceback.format_exc(),
            }
            results.append(entry)
            logger.warning("  ✘ FAILED: %s", e)

        # Periodic garbage collection to keep memory stable
        if idx % 5 == 0:
            gc.collect()

    t_sweep_total = time.perf_counter() - t_sweep_start

    # ── Rank results ───────────────────────────────────────────────────
    ranked = sorted(
        [r for r in results if r["composite_score"] is not None],
        key=lambda r: r["composite_score"],
        reverse=True,
    )

    # ── Save sweep JSON ────────────────────────────────────────────────
    sweep_data = {
        "sweep_timestamp": _timestamp(),
        "sweep_duration_seconds": round(t_sweep_total, 1),
        "total_combinations": total,
        "successful": sum(1 for r in results if r["status"] == "success"),
        "failed": sum(1 for r in results if r["status"] == "failed"),
        "parameter_grid": {
            "diffusion_steps": DIFFUSION_STEPS,
            "inference_cfg_rate": INFERENCE_CFG_RATE,
            "length_adjust": LENGTH_ADJUST,
        },
        "weights": {
            "mel_distance": 0.30,
            "pitch_similarity": 0.20,
            "energy_preservation": 0.15,
            "duration_preservation": 0.10,
            "spectral_flux": 0.15,
            "snr_estimate": 0.10,
        },
        "results": results,
        "ranking": [
            {
                "rank": rank,
                "combo_label": r["combo_label"],
                "composite_score": r["composite_score"],
                "parameters": r["parameters"],
                "latency_ms": r["latency_ms"],
                "rtf": r["rtf"],
            }
            for rank, r in enumerate(ranked, 1)
        ],
    }

    with open(sweep_file, "w") as f:
        json.dump(sweep_data, f, indent=2, default=str)
    logger.info("Sweep results saved: %s", sweep_file)

    # ── Print ranked table ─────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"PARAMETER SWEEP RESULTS  —  {total} combinations, {t_sweep_total:.1f}s total")
    print("=" * 70)

    if ranked:
        print(f"\n{'Rank':<5} {'Label':<35} {'Score':>7} {'Latency':>10} {'RTF':>8}")
        print(f"  {'─'*4} {'─'*35} {'─'*7} {'─'*10} {'─'*8}")
        for rank, r in enumerate(ranked, 1):
            rtf_str = f"{r['rtf']:.3f}" if r["rtf"] is not None else "  N/A"
            print(
                f"  {rank:<4} {r['combo_label']:<35} "
                f"{r['composite_score']:>7.2f} "
                f"{r['latency_ms']:>9.1f}ms "
                f"{rtf_str:>8}"
            )

        # Top 5 detail
        print(f"\n{'─'*70}")
        print("TOP 5 PARAMETER COMBINATIONS")
        print("─" * 70)
        for rank, r in enumerate(ranked[:5], 1):
            p = r["parameters"]
            m = r["metrics"]
            print(f"\n  #{rank}: {r['combo_label']}  —  Score: {r['composite_score']:.2f}")
            print(f"       diffusion_steps={p['diffusion_steps']}, "
                  f"cfg_rate={p['inference_cfg_rate']}, "
                  f"length_adjust={p['length_adjust']}")
            # Key metrics
            mel_v = m.get("mel_spectrogram_distance", {}).get("value", float("nan"))
            pit_v = m.get("pitch_similarity", {}).get("value", float("nan"))
            snr_v = m.get("snr_estimate", {}).get("value", float("nan"))
            print(f"       mel_dist={mel_v:.4f}, pitch_sim={pit_v:.4f}, "
                  f"snr={snr_v:.1f}dB, latency={r['latency_ms']:.0f}ms")
    else:
        print("  No successful conversions.")

    # Failed combinations
    failed = [r for r in results if r["status"] == "failed"]
    if failed:
        print(f"\n  Failed ({len(failed)}):")
        for r in failed:
            print(f"    {r['combo_label']}: {r['error']}")

    print("\n" + "=" * 70)
    print(f"Report: {sweep_file}")
    print(f"Audio:  {AUDIO_DIR}")
    print("=" * 70)

    # ── Cleanup ────────────────────────────────────────────────────────
    converter.unload()
    gc.collect()

    return sweep_file


def main():
    parser = argparse.ArgumentParser(
        description="Run parameter sweep for Seed-VC quality optimization."
    )
    parser.add_argument(
        "--max-combos",
        type=int,
        default=None,
        help="Limit number of combinations (full grid is 36, ~24min at 40s each).",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("SEED-VC PARAMETER SWEEP")
    print("=" * 70)
    print(f"Source:      {SOURCE_PATH}")
    print(f"Reference:   {REFERENCE_PATH}")
    print(f"Device:      cpu")
    print(f"Output dir:  {SWEEP_DIR}")
    print()

    if args.max_combos is None:
        logger.info("Full grid: %d × %d × %d = %d combinations (%s estimated)",
                    len(DIFFUSION_STEPS), len(INFERENCE_CFG_RATE),
                    len(LENGTH_ADJUST),
                    len(DIFFUSION_STEPS) * len(INFERENCE_CFG_RATE) * len(LENGTH_ADJUST),
                    f"~{(len(DIFFUSION_STEPS) * len(INFERENCE_CFG_RATE) * len(LENGTH_ADJUST)) * 40 / 60:.0f}min")
        resp = input("Proceed with all combinations? [Y/n]: ").strip().lower()
        if resp == "n":
            max_c = input("Enter max combinations: ").strip()
            try:
                args.max_combos = int(max_c)
            except ValueError:
                logger.error("Invalid input, exiting.")
                sys.exit(1)

    sweep_file = run_sweep(args.max_combos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
