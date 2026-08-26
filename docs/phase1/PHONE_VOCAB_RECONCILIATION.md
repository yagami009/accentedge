# Phone Vocabulary Reconciliation — AccentEdge Phase 1

## The Mismatch

| Component | Expected size | Actual size | Status |
|-----------|--------------|-------------|--------|
| Denoiser `phone_emb` embedding table | 393 rows (`phone_vocab_size=393`) | 393 | Correct (from paper) |
| PhonemePipeline `_PHONEME_LIST` | — | 93 entries (IDs 0–92) | Subset of 393 |
| Denoiser `padding_idx` | 392 (last row) | 392 | Correct (from paper) |
| Pipeline `PAD_ID` | ID 1 (`'sp'`) | 1 | Semantic padding |

## Why 393 ≠ 93 Active Symbols

The denoiser's `phone_vocab_size=393` comes directly from the FAC-FACodec paper
(arxiv:2510.10785v2, Table 3). The paper uses a **multilingual** setup: eSpeak-ng
phonemized across multiple languages, where IPA symbols with stress markers,
diacritics, and language-specific variants accumulate to roughly 393 distinct
symbols.

AccentEdge Phase 1 is **English-only** (LJSpeech dataset), so the phonemizer
only produces a small subset of those 393 possible symbols. Our
`_PHONEME_LIST` contains 93 IPA symbols that eSpeak-ng emits for English.

This is common in multilingual models: the embedding table is sized for the
full multilingual vocabulary, but a monolingual deployment only uses a subset.
The unused embedding rows simply never receive gradient updates.

## Chosen Resolution: Option B — Keep 393, Use Subset

**Decision**: Preserve the denoiser's 393-row embedding table (paper-faithful).
The PhonemePipeline produces IDs in range 0–92, all safely within 0–392.

### Why not the other options?

- **Option A (expand vocabulary)**: We would need to enumerate all ~393
  multilingual IPA symbols from eSpeak-ng across all supported languages.
  This is unnecessary overhead for an English-only system and would bloat
  the alignment model's output space.

- **Option C (resize denoiser embedding)**: Changing `phone_vocab_size` from
  393 to 93 would break compatibility with any pretrained denoiser weights
  loaded from the paper's architecture. It also makes future multilingual
  extension harder.

## Implementation Details

### PhonemePipeline (`src/accentedge/phase1/phoneme_pipeline.py`)

1. Added `PHONE_VOCAB_SIZE = 393` constant — must match denoiser's
   `phone_vocab_size`.
2. Added vocabulary size contract comment block documenting the relationship
   between 393 (embedding capacity) and 93 (active English symbols).
3. Cleaned up `_PHONEME_LIST`:
   - Removed duplicate `'ɜː'` entry (present at both index 40 and 42 in original)
   - Removed corrupted U+FFFD characters (file encoding corruption)
   - Restored missing `'ø'` (index 13) and `'ɜ̃'` (index 32) that were lost
     to the corruption
   - Final count: 93 clean entries, no duplicates
4. All pipeline IDs remain in range 0–92, which is < 393.

### Denoiser (`src/accentedge/phase1/denoiser.py`)

**No changes.** The 393-row embedding table is preserved exactly as specified
by the paper. The 300 unused rows (IDs 93–391) simply never get addressed by
the pipeline's output.

### Key Distinction: padding_idx=392 vs pad_id=1

The denoiser uses `padding_idx=392` — the last row of the embedding table,
always zeroed by PyTorch. This is for PyTorch's internal padding semantics
(when `phone_ids` contains the value 392, the embedding is zeroed).

The pipeline uses `pad_id=1` (the `'sp'` silence symbol) — this is an
**active** phoneme ID that represents silence/no-phoneme frames in the output
tensor. It is NOT the same as the PyTorch padding_idx.

**These two padding mechanisms are independent and serve different purposes.**

## Changes Required

| File | Change |
|------|--------|
| `src/accentedge/phase1/phoneme_pipeline.py` | Added `PHONE_VOCAB_SIZE` constant, vocabulary contract comment, fixed duplicate/corrupted IPA entries |
| `tests/unit/test_phoneme_pipeline.py` | Added phone vocab size contract tests |

## Verification

Run the phone vocab contract tests:

```bash
python -m pytest tests/unit/test_phoneme_pipeline.py -v -k "vocab"
```

These tests verify:
1. `PHONE_VOCAB_SIZE == 393` (matches denoiser)
2. All pipeline IDs are < 393
3. No duplicate symbols in the vocabulary
4. `PAD_ID` (1) ≠ denoiser `padding_idx` (392)
5. `_PHONEME_LIST` contains 93 entries
