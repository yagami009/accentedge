#!/usr/bin/env python3
"""
Record test audio samples for Gate 1B testing.

Records 5 sentences in Indian English from the microphone.
Saves to data/samples/ for use in the prototype.
"""

import argparse
import sys
import time
from pathlib import Path

try:
    import sounddevice as sd
    import soundfile as sf
    import numpy as np
except ImportError:
    print("ERROR: sounddevice not installed.")
    print("Install with: pip install sounddevice soundfile")
    sys.exit(1)


SENTENCES = [
    "Hello, how can I help you today?",
    "Thank you for calling our support line.",
    "I understand your concern, let me look into that.",
    "Could you please provide your account number?",
    "Is there anything else I can assist you with?",
]


def record_sentence(duration: float = 5.0, sample_rate: int = 16000) -> np.ndarray:
    """Record audio from microphone."""
    print(f"  Recording {duration}s... (speak now)")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio


def main():
    parser = argparse.ArgumentParser(description="Record test audio samples")
    parser.add_argument("--output-dir", default="data/samples", help="Output directory")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration per sentence")
    parser.add_argument("--sample-rate", type=int, default=16000)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Test Audio Recorder")
    print("=" * 60)
    print(f"Output: {output_dir}")
    print(f"Duration per sentence: {args.duration}s")
    print(f"Sample rate: {args.sample_rate}Hz")
    print()
    print("You will record 5 sentences in Indian English.")
    print("Speak naturally, as if you're a BPO agent on a call.")
    print()

    for i, sentence in enumerate(SENTENCES):
        print(f"\n--- Sentence {i+1}/5 ---")
        print(f"Text: \"{sentence}\"")
        print("Press Enter when ready...")
        input()

        audio = record_sentence(duration=args.duration, sample_rate=args.sample_rate)

        # Save
        output_path = output_dir / f"en_in_sentence_{i+1:02d}.wav"
        sf.write(str(output_path), audio, args.sample_rate)
        print(f"  Saved: {output_path}")

        # Small pause between sentences
        if i < len(SENTENCES) - 1:
            time.sleep(1)

    print("\n" + "=" * 60)
    print("Recording complete!")
    print(f"Files saved to: {output_dir}")
    print()
    print("Next steps:")
    print("  1. Run Gate 1A: python scripts/gate_1a_whisper.py --audio data/samples/en_in_sentence_01.wav")
    print("  2. Get a reference voice (en-US audio) for XTTS v2")
    print("  3. Run Gate 1B: python scripts/gate_1b_xtts.py --input ... --reference ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
