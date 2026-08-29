#!/usr/bin/env python3
"""
Gate 1C — XTTS v2 end-to-end latency test.

Since XTTS v2 bundles its own vocoder, this test measures the full
Stage 2 latency (text → audio) which is what actually matters for the pipeline.

Pass criteria: <5s for a single sentence (RTF < 2x for 2-3s output)
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converter import AccentConverter


def main():
    parser = argparse.ArgumentParser(description="Gate 1C: XTTS v2 end-to-end latency")
    parser.add_argument("--reference", required=True, help="Reference audio for voice style")
    parser.add_argument("--text", default="Hello, how can I help you today?")
    parser.add_argument("--output", default="data/samples/gate1c_output.wav")
    parser.add_argument("--runs", type=int, default=3)
    args = parser.parse_args()

    ref_path = Path(args.reference)
    if not ref_path.exists():
        print(f"ERROR: Reference audio not found: {ref_path}")
        sys.exit(1)

    print("=" * 60)
    print("GATE 1C: XTTS v2 End-to-End Latency")
    print("=" * 60)
    print(f"Reference: {ref_path}")
    print(f"Text: {args.text}")
    print(f"Runs: {args.runs}")
    print()

    converter = AccentConverter()

    latencies = []
    for i in range(args.runs):
        t0 = time.time()
        result = converter.convert(
            text=args.text,
            reference_audio=str(ref_path),
            output_path=f"data/samples/gate1c_run_{i}.wav",
        )
        elapsed = time.time() - t0
        latencies.append(elapsed)
        print(f"  Run {i+1}: {elapsed:.2f}s")

    avg = sum(latencies) / len(latencies)
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Avg latency: {avg:.2f}s")
    print(f"Min: {min(latencies):.2f}s, Max: {max(latencies):.2f}s")

    if avg < 5.0:
        print("✓ PASS: XTTS v2 is fast enough for batch demo use")
    else:
        print("✗ SLOW: Consider smaller model or quantized version for demo")
