#!/usr/bin/env python3
"""
Gate 1A — Whisper latency test.

Measures Whisper small inference speed on M1 Pro.
Pass criteria: RTF < 0.5x (faster than real-time)

Usage:
    python scripts/gate_1a_whisper.py --audio data/samples/test.wav
    python scripts/gate_1a_whisper.py --audio data/samples/test.wav --model tiny
"""

import argparse
import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.encoder import SpeechEncoder


def main():
    parser = argparse.ArgumentParser(description="Gate 1A: Whisper latency test")
    parser.add_argument("--audio", required=True, help="Path to test audio file")
    parser.add_argument("--model", default="small", choices=["tiny", "base", "small", "medium"])
    parser.add_argument("--runs", type=int, default=3, help="Number of runs to average")
    args = parser.parse_args()

    audio_path = Path(args.audio)
    if not audio_path.exists():
        print(f"ERROR: Audio file not found: {audio_path}")
        print("Record a test file first:")
        print("  python scripts/record_test.py")
        sys.exit(1)

    print("=" * 60)
    print("GATE 1A: Whisper Latency Test")
    print("=" * 60)
    print(f"Audio: {audio_path}")
    print(f"Model: {args.model}")
    print(f"Runs: {args.runs}")
    print()

    # Load encoder
    encoder = SpeechEncoder(model_size=args.model)

    # Warm-up run (first run is always slower due to JIT)
    print("\n--- Warm-up run ---")
    encoder.transcribe(str(audio_path))

    # Timed runs
    print(f"\n--- {args.runs} timed runs ---")
    stats = encoder.get_latency_stats(str(audio_path), runs=args.runs)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Model: {stats['model']}")
    print(f"Runs: {stats['runs']}")
    print()
    print(f"Latency (seconds):")
    print(f"  Min:  {stats['latency_min']:.3f}s")
    print(f"  Avg:  {stats['latency_avg']:.3f}s")
    print(f"  Max:  {stats['latency_max']:.3f}s")
    print()
    print(f"Real-time factor (RTF):")
    print(f"  Min:  {stats['rtf_min']:.3f}x")
    print(f"  Avg:  {stats['rtf_avg']:.3f}x")
    print(f"  Max:  {stats['rtf_max']:.3f}x")
    print()

    # Pass/fail
    target_rtf = 0.5
    if stats["rtf_avg"] < target_rtf:
        print(f"✓ PASS: RTF {stats['rtf_avg']:.3f}x < {target_rtf}x")
        print("  Whisper small is fast enough for the pipeline.")
    else:
        print(f"✗ FAIL: RTF {stats['rtf_avg']:.3f}x > {target_rtf}x")
        print(f"  Consider downgrading to 'tiny' or 'base' model.")
        print(f"  Or optimize: use CoreML export, reduce batch size.")

    print("=" * 60)


if __name__ == "__main__":
    main()
