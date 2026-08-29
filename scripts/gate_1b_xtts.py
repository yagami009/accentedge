#!/usr/bin/env python3
"""
Gate 1B — XTTS v2 accent conversion test.

THE MAKE-OR-BREAK TEST.
Tests whether XTTS v2 can convert en-IN speech to en-US accent
while preserving speaker identity.

Usage:
    python scripts/gate_1b_xtts.py \
        --input data/samples/en_in_speaker.wav \
        --text "Hello, how can I help you today?" \
        --reference data/samples/en_us_reference.wav \
        --output data/samples/gate1b_output.wav
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.converter import AccentConverter


def main():
    parser = argparse.ArgumentParser(description="Gate 1B: XTTS v2 accent conversion test")
    parser.add_argument("--input", required=True, help="Path to en-IN input audio (for transcription)")
    parser.add_argument("--text", required=True, help="Text to synthesize in target accent")
    parser.add_argument("--reference", required=True, help="Path to reference audio (target voice style)")
    parser.add_argument("--output", default="data/samples/gate1b_output.wav", help="Output path")
    parser.add_argument("--model", default="tts_models/multilingual/multi-dataset/xtts_v2")
    args = parser.parse_args()

    input_path = Path(args.input)
    ref_path = Path(args.reference)
    output_path = Path(args.output)

    for p in [input_path, ref_path]:
        if not p.exists():
            print(f"ERROR: File not found: {p}")
            sys.exit(1)

    print("=" * 60)
    print("GATE 1B: XTTS v2 Accent Conversion Test")
    print("=" * 60)
    print(f"Input audio:  {input_path}")
    print(f"Reference:    {ref_path}")
    print(f"Text:         {args.text}")
    print(f"Output:       {output_path}")
    print(f"Model:        {args.model}")
    print()
    print("NOTE: This is the make-or-break test.")
    print("Listen to the output and judge:")
    print("  1. Does it sound like the SAME speaker?")
    print("  2. Does it sound like AMERICAN accent (not Indian)?")
    print("  3. Is it intelligible and natural?")
    print("=" * 60)

    # Load converter
    converter = AccentConverter(model_name=args.model)

    # Run conversion
    t0 = time.time()
    result = converter.convert(
        text=args.text,
        reference_audio=str(ref_path),
        output_path=str(output_path),
        language="en",
    )
    total_latency = time.time() - t0

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Output:       {result['output_path']}")
    print(f"Latency:      {result['latency']:.2f}s")
    print(f"Total time:   {total_latency:.2f}s")
    print()
    print("JUDGMENT:")
    print("  [ ] Same speaker identity preserved?")
    print("  [ ] American accent (not Indian)?")
    print("  [ ] Intelligible and natural?")
    print()
    print("If all three are YES → LOCKED. Build cascaded pipeline.")
    print("If voice changed entirely → Fine-tuning required.")
    print("If broken/unintelligible → Pivot approach.")
    print("=" * 60)


if __name__ == "__main__":
    main()
