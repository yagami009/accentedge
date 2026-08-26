# Transcript Dependency Analysis

## Verdict: TRANSCRIPTS REQUIRED for paper-faithful inference

### Training

The FAC-FACodec denoiser is conditioned on phoneme embeddings $\pi$ (extracted via phonemizer + eSpeak-ng) aligned with Wav2Vec2 XLSR. Training requires phoneme sequences for each utterance.

Status: **REQUIRED** (PAPER_EXPLICIT)

### Inference

The denoiser $s_\theta(x_t, t, \pi)$ takes phoneme embeddings $\pi$ as conditioning input via FiLM and additive embeddings. Without transcript → phoneme extraction → alignment, the denoiser has no conditioning signal.

Status: **REQUIRED** (PAPER_EXPLICIT)

### Component

- Phoneme extraction: phonemizer + eSpeak-ng
- Alignment: Wav2Vec2 XLSR
- Conditioning: FiLM layers + additive embeddings in 6-layer Transformer denoiser

### Could it be replaced?

In principle, phonemes could be extracted from the source speech via:
- A phoneme recognizer (e.g., Wav2Vec2 XLS-R fine-tuned for phoneme recognition)
- An ASR system

But this is:
1. An additional model dependency
2. Not the paper method
3. Potentially circular (accented speech → phoneme recognizer may misrecognize)

### Structural?

Yes. The phoneme conditioning is structural to the published method. Removing it changes the scientific experiment.

### Phase 1 Decision

Implement paper-faithful transcript/phoneme conditioning. Create separate experimental mode `experimental_unconditioned` later if needed.
