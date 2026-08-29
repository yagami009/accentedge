#!/usr/bin/env python3
"""
Evaluation runner — finds the latest successful conversion and runs
audio quality evaluation, printing a human-readable summary and saving
JSON results to results/evaluation/.
"""

import logging
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluator import AudioEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def find_latest_successful_run(results_dir: Path):
    """Find the latest run directory with source, output, and reference WAVs."""
    if not results_dir.exists():
        return None
    run_dirs = sorted(results_dir.iterdir(), key=lambda d: d.name)
    for run_dir in reversed(run_dirs):
        if not run_dir.is_dir():
            continue
        has_source = (run_dir / "source.wav").exists()
        has_output = (run_dir / "output.wav").exists()
        has_reference = (run_dir / "reference.wav").exists()
        if has_source and has_output and has_reference:
            return run_dir
    return None


def main():
    """Run evaluation on the latest successful conversion."""
    offline_dir = PROJECT_ROOT / "results" / "offline"
    eval_dir = PROJECT_ROOT / "results" / "evaluation"

    run_dir = find_latest_successful_run(offline_dir)
    if run_dir is None:
        logger.error("No successful conversion run found in %s", offline_dir)
        sys.exit(1)

    run_id = run_dir.name
    logger.info("Evaluating run: %s", run_id)

    # Load audio files
    source, sr_src = sf.read(str(run_dir / "source.wav"))
    output, sr_out = sf.read(str(run_dir / "output.wav"))
    reference, sr_ref = sf.read(str(run_dir / "reference.wav"))

    # Run evaluation
    evaluator = AudioEvaluator(sample_rate=int(sr_src), output_dir=str(eval_dir))
    report = evaluator.evaluate_and_save(source, output, reference=reference, run_id=run_id)

    # Print human-readable summary
    evaluator.print_report(report)

    # Quick scorecard
    m = report.get("metrics", {})
    print("\nQuick Scorecard:")
    print("-" * 60)
    print(f"  Mel distance:     {m.get('mel_spectrogram_distance', {}).get('value', 0):.4f}")
    print(f"  Pitch corr:       {m.get('pitch_similarity', {}).get('value', 0):.4f}")
    print(f"  Energy ratio:     {m.get('energy_preservation', {}).get('value', 0):.4f}")
    print(f"  Duration ratio:   {m.get('duration_preservation', {}).get('value', 0):.4f}")
    print(f"  Spectral flux:    {m.get('spectral_flux', {}).get('value', 0):.4f}")
    print("-" * 60)
    print(f"\nReport saved: {report.get('saved_to', 'N/A')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
