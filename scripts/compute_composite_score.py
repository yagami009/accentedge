#!/usr/bin/env python3
"""
Compute a weighted composite quality score from evaluation metrics.

Normalizes individual metrics to a 0-100 scale and combines them using
predefined weights. Handles NaN/inf values gracefully.

Usage:
    python scripts/compute_composite_score.py <metrics.json>
    python scripts/compute_composite_score.py  # reads last sweep file
"""

import argparse
import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── Metric weights (sum to 1.0) ────────────────────────────────────────
WEIGHTS = {
    "mel_distance":        0.30,   # lower is better
    "pitch_similarity":    0.20,   # higher is better
    "energy_preservation": 0.15,   # closer to 1.0
    "duration_preservation": 0.10, # closer to 1.0
    "spectral_flux":       0.15,   # lower is better
    "snr_estimate":        0.10,   # higher is better
}

# ── Normalization parameters ──────────────────��────────────────────────
# For "lower is better" metrics: score = clamp((1 - val / max_val) * 100)
# For "higher is better": score = clamp((val - min_val) / (max_val - min_val) * 100)
# For "closer to target": score = clamp((1 - |val - target| / max_dev) * 100)

MEL_DISTANCE_MAX    = 3.0   # L2 distance where score drops to 0
SPECTRAL_FLUX_MAX   = 3.0   # flux where score drops to 0
ENERGY_TARGET       = 1.0   # ideal ratio
ENERGY_MAX_DEV      = 2.0   # deviation where score drops to 0
DURATION_TARGET     = 1.0   # ideal ratio
DURATION_MAX_DEV    = 0.5   # deviation where score drops to 0
SNR_MIN             = -10.0 # SNR dB where score is 0
SNR_MAX             = 30.0  # SNR dB where score is 100
PITCH_MIN           = -1.0  # Pearson r minimum (maps to 0)
PITCH_MAX           = 1.0   # Pearson r maximum (maps to 100)


def _safe_float(val) -> float:
    """Extract a float from a metric entry, return NaN on failure."""
    if isinstance(val, dict):
        val = val.get("value", float("nan"))
    if val is None:
        return float("nan")
    try:
        f = float(val)
        return f if math.isfinite(f) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _clamp(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, v))


def _nan_score(val: float, fallback: float) -> float:
    """Return fallback if NaN, else val."""
    return fallback if math.isnan(val) else val


def compute_composite_score(metrics: dict) -> float:
    """
    Compute weighted composite quality score from an evaluation metrics dict.

    The metrics dict should contain the keys produced by AudioQualityMetrics.compute_all():
      - mel_spectrogram_distance
      - pitch_similarity
      - energy_preservation
      - duration_preservation
      - spectral_flux
      - snr_estimate

    Each value can be a raw float or a MetricResult.to_dict() entry
    (with a 'value' key).

    Returns:
        float in [0, 100]
    """
    # --- Extract raw values ---
    mel_dist   = _safe_float(metrics.get("mel_spectrogram_distance"))
    pitch_sim  = _safe_float(metrics.get("pitch_similarity"))
    energy_pres = _safe_float(metrics.get("energy_preservation"))
    dur_pres   = _safe_float(metrics.get("duration_preservation"))
    spec_flux  = _safe_float(metrics.get("spectral_flux"))
    snr        = _safe_float(metrics.get("snr_estimate"))

    # --- Normalize to 0-100 sub-scores ---

    # Mel distance: lower is better. 0→100, 3.0→0
    mel_score = _clamp((1.0 - _nan_score(mel_dist, MEL_DISTANCE_MAX) / MEL_DISTANCE_MAX) * 100.0)

    # Pitch similarity: Pearson r, -1→0, 0→50, 1→100
    pitch_score = _clamp((_nan_score(pitch_sim, PITCH_MIN) - PITCH_MIN) / (PITCH_MAX - PITCH_MIN) * 100.0)

    # Energy preservation: closer to 1.0 is better
    energy_dev = abs(_nan_score(energy_pres, ENERGY_TARGET + ENERGY_MAX_DEV) - ENERGY_TARGET)
    energy_score = _clamp((1.0 - energy_dev / ENERGY_MAX_DEV) * 100.0)

    # Duration preservation: closer to 1.0 is better
    dur_dev = abs(_nan_score(dur_pres, DURATION_TARGET + DURATION_MAX_DEV) - DURATION_TARGET)
    dur_score = _clamp((1.0 - dur_dev / DURATION_MAX_DEV) * 100.0)

    # Spectral flux: lower is better. 0→100, 3.0→0
    flux_score = _clamp((1.0 - _nan_score(spec_flux, SPECTRAL_FLUX_MAX) / SPECTRAL_FLUX_MAX) * 100.0)

    # SNR estimate: higher is better. -10→0, 30→100
    snr_val = _nan_score(snr, SNR_MIN)
    snr_score = _clamp((snr_val - SNR_MIN) / (SNR_MAX - SNR_MIN) * 100.0)

    # --- Weighted composite ---
    composite = (
        WEIGHTS["mel_distance"]        * mel_score
        + WEIGHTS["pitch_similarity"]  * pitch_score
        + WEIGHTS["energy_preservation"] * energy_score
        + WEIGHTS["duration_preservation"] * dur_score
        + WEIGHTS["spectral_flux"]     * flux_score
        + WEIGHTS["snr_estimate"]      * snr_score
    )

    return round(composite, 2)


def score_from_file(path: str) -> float:
    """Load a metrics JSON file and compute its composite score."""
    with open(path, "r") as f:
        data = json.load(f)
    metrics = data.get("metrics", data)
    return compute_composite_score(metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Compute composite quality score from evaluation metrics."
    )
    parser.add_argument(
        "metrics_file",
        nargs="?",
        default=None,
        help="Path to a metrics JSON file (e.g. results/sweep/sweep_*.json). "
             "If omitted, scans results/sweep/ for the latest file.",
    )
    args = parser.parse_args()

    if args.metrics_file:
        metrics_path = Path(args.metrics_file)
    else:
        sweep_dir = PROJECT_ROOT / "results" / "sweep"
        candidates = sorted(sweep_dir.glob("sweep_*.json"), key=lambda p: p.name)
        if not candidates:
            print("No sweep files found in results/sweep/", file=sys.stderr)
            sys.exit(1)
        metrics_path = candidates[-1]

    if not metrics_path.exists():
        print(f"File not found: {metrics_path}", file=sys.stderr)
        sys.exit(1)

    score = score_from_file(str(metrics_path))
    print(f"Composite quality score: {score:.2f} / 100")
    print(f"  Source: {metrics_path}")

    # Show per-metric breakdown if available
    with open(metrics_path, "r") as f:
        data = json.load(f)

    # Handle both single-metrics dict and sweep results
    if "results" in data:
        # Sweep file — show top 5 scores
        results = sorted(
            data["results"],
            key=lambda r: r.get("composite_score", 0) or 0,
            reverse=True,
        )
        print(f"\nTop 5 from {data.get('total_combinations', '?')} combinations:")
        print(f"  {'Rank':<5} {'Label':<35} {'Score':>7}")
        print(f"  {'─'*5} {'─'*35} {'─'*7}")
        for rank, r in enumerate(results[:5], 1):
            score = r.get("composite_score", 0) or 0
            print(f"  {rank:<5} {r['combo_label']:<35} {score:>7.2f}")
    elif "metrics" in data:
        # Single evaluation report
        metrics = data["metrics"]
        print("\nPer-metric contributions:")
        name_map = {
            "mel_spectrogram_distance": "Mel Distance (w=0.30)",
            "pitch_similarity": "Pitch Similarity (w=0.20)",
            "energy_preservation": "Energy Preservation (w=0.15)",
            "duration_preservation": "Duration Preservation (w=0.10)",
            "spectral_flux": "Spectral Flux (w=0.15)",
            "snr_estimate": "SNR Estimate (w=0.10)",
        }
        raw = {}
        for key, label in name_map.items():
            v = _safe_float(metrics.get(key, {}))
            raw[label] = v

        # Recompute each sub-score for display
        mel_s    = _clamp((1.0 - _nan_score(raw["Mel Distance (w=0.30)"], 3.0) / 3.0) * 100)
        pitch_s  = _clamp((_nan_score(raw["Pitch Similarity (w=0.20)"], -1.0) + 1.0) / 2.0 * 100)
        e_dev    = abs(_nan_score(raw["Energy Preservation (w=0.15)"], 3.0) - 1.0)
        energy_s = _clamp((1.0 - e_dev / 2.0) * 100)
        d_dev    = abs(_nan_score(raw["Duration Preservation (w=0.10)"], 3.0) - 1.0)
        dur_s    = _clamp((1.0 - d_dev / 0.5) * 100)
        flux_s   = _clamp((1.0 - _nan_score(raw["Spectral Flux (w=0.15)"], 3.0) / 3.0) * 100)
        snr_v    = _nan_score(raw["SNR Estimate (w=0.10)"], -10.0)
        snr_s    = _clamp((snr_v + 10.0) / 40.0 * 100)

        subs = [
            ("Mel Distance (w=0.30)",        mel_s,    raw["Mel Distance (w=0.30)"]),
            ("Pitch Similarity (w=0.20)",    pitch_s,  raw["Pitch Similarity (w=0.20)"]),
            ("Energy Preservation (w=0.15)", energy_s, raw["Energy Preservation (w=0.15)"]),
            ("Duration Preservation (w=0.10)", dur_s,  raw["Duration Preservation (w=0.10)"]),
            ("Spectral Flux (w=0.15)",       flux_s,   raw["Spectral Flux (w=0.15)"]),
            ("SNR Estimate (w=0.10)",        snr_s,    raw["SNR Estimate (w=0.10)"]),
        ]
        for label, sub_score, raw_val in subs:
            raw_str = f"{raw_val:.4f}" if math.isfinite(raw_val) else str(raw_val)
            print(f"  {label}: {sub_score:6.2f}  (raw={raw_str})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
