#!/usr/bin/env python3
"""Record Phase-0 source utterances."""

import sounddevice as sd
import soundfile as sf
import numpy as np
from pathlib import Path

SAMPLE_RATE = 16000
CHANNELS = 1
OUTPUT_DIR = Path("/Users/ayushmh/accentedge/phase0/recordings")

SENTENCES = [
    "Could you please confirm your account number and the billing address on file?",
    "I can see a charge of thirty dollars posted on the thirteenth of August.",
    "Let me check the details and get back to you in about fifteen minutes.",
    "A as in Alpha, B as in Bravo, seven, one, three, nine, K.",
    "She thought the three tickets were worth thirty pounds, but they cost thirty-three.",
]


def record_utterance(duration_seconds: float, label: str) -> np.ndarray:
    print(f"\n{'='*60}")
    print(f"RECORDING: {label}")
    print(f"{'='*60}")
    print(f"Get ready... starting in 3 seconds.")
    sd.sleep(3000)
    print(f"Recording NOW ({duration_seconds}s)... speak naturally")
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=CHANNELS,
        dtype=np.float32,
    )
    sd.wait()
    print(f"Done.")
    return audio


def save_audio(audio: np.ndarray, filename: str):
    output_path = OUTPUT_DIR / filename
    sf.write(str(output_path), audio, SAMPLE_RATE)
    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("="*60)
    print("PHASE-0 RECORDING SESSION")
    print("="*60)
    print(f"Sample rate: {SAMPLE_RATE} Hz")
    print(f"Channels: {CHANNELS} (mono)")
    print(f"Output: {OUTPUT_DIR}")
    print(f"\nAvailable audio devices:")
    print(sd.query_devices())
    input("\nPress ENTER when ready to start...")

    # Record 5 scripted sentences
    for i, sentence in enumerate(SENTENCES, 1):
        print(f"\n>>> Say this sentence:")
        print(f"    \"{sentence}\"")
        audio = record_utterance(8.0, f"Sentence {i}")
        save_audio(audio, f"source_00{i}.wav")

    # Record spontaneous speech
    print(f"\n{'='*60}")
    print("SPONTANEOUS SPEECH")
    print(f"{'='*60}")
    print(f"Talk naturally for 20-30 seconds about anything.")
    print(f"Your day, work, what you had for lunch, whatever.")
    audio = record_utterance(30.0, "Spontaneous speech")
    save_audio(audio, "source_006.wav")

    print(f"\n{'='*60}")
    print("ALL RECORDINGS COMPLETE")
    print(f"{'='*60}")
    print(f"Files saved to: {OUTPUT_DIR}")
    for f in sorted(OUTPUT_DIR.glob("*.wav")):
        info = sf.info(str(f))
        print(f"  {f.name}: {info.frames/SAMPLE_RATE:.1f}s")


if __name__ == "__main__":
    main()
