# TGFP v2 — Target Generation Feasibility Protocol

**Version:** 2.0
**Date:** 2026-08-25
**Status:** Draft

---

## Purpose

TGFP defines the controlled experiment that determines whether AccentEdge can construct a usable accent transformation target — before any model training begins.

TGFP v1 was exploratory. TGFP v2 is precise: it specifies a single first experiment (Step 0) whose result determines whether the project proceeds to Phase 1 or stops.

---

## Core Question

> Can we produce a target waveform that sounds like the **same human speaking differently**, not like a different voice or impression?

---

## Step 0 — Single-Speaker Feasibility Test

### Setup

- **One speaker:** Indian English, self-identifies as having a noticeable Indian accent
- **One sentence:** Chosen to contain at least one phoneme contrast between en-IN and en-US (e.g., /t/ flapping, /æ/ raising, /ɪ/ tensing, /oʊ/ monophthongization)
- **One strategy:** Strategy B (hand-built target)

### Strategy B — Hand-Built Target

1. Record source sentence: Indian-English pronunciation
2. Expert (native US English speaker) produces a target pronunciation guide:
   - Phone-level transcription of US-neutral realization
   - Word-level stress pattern
   - Phrase-level intonation contour
3. Source speaker attempts to produce the target pronunciation while maintaining their own voice
4. Record multiple takes
5. Select best take as `strategy_b/candidate.wav`

### Evaluation

Listeners (n >= 3, native US English, trained in accent evaluation) rate:

| Dimension | Scale |
|---|---|
| Same speaker? | 1–5 (1 = different person, 5 = same person) |
| Accent shift? | 1–5 (1 = still Indian, 5 = neutral US) |
| Naturalness? | 1–5 (1 = robotic, 5 = natural) |
| Content preserved? | Yes / No / Partial |

### Pass Criteria (all must be met)

- Same speaker >= 4/5
- Accent shift >= 3/5
- Naturalness >= 3/5
- Content preserved = Yes

### Decision Tree

```
STEP 0 RESULT
       │
   ┌───┴────┐
   │        │
   PASS     FAIL
   │        │
   ▼        ▼
Gate 1A   Fundamental
           fail
```

If Step 0 passes: proceed to Gate 1A (corpus expansion, natural gold, alignment verification).

If Step 0 fails: investigate whether failure is due to speaker inability, strategy, or fundamental acoustic constraints. Do NOT proceed to model training until the failure mode is understood.

---

## Gate 1A — Corpus and Alignment

Only after Step 0 passes.

1. Expand to ~30 utterances per speaker
2. Add 3-5 speakers
3. Record natural cross-accent pairs (same speaker, IN then US rendering)
4. Phone-level forced alignment
5. Manual correction of alignment in pronunciation-critical regions
6. Per-token labels: ALREADY-TARGET / DEVIANT / AMBIGUOUS

---

## Gate 1B — Strategy Comparison

Compare Strategy A, B, C on the expanded corpus.

Winner determined by combined score across all evaluation dimensions.

---

## Gate 2 — Natural Gold Validation

Verify that natural cross-accent recordings show the same identity-preservation pattern as Strategy B.

If natural gold fails but synthetic strategies pass, the model must learn a transformation that humans cannot naturally produce — a much harder problem.

---

## Gate 3 — Model Feasibility

First model attempt reproduces Step 0 result on held-out utterances.

If model cannot match Strategy B quality on single utterances, do not proceed to streaming.

---

## Phase 1 — BPO Benchmark

Only after Gate 3 passes.

Expand to BPO-specific content: numbers, names, addresses, fast speech, interruptions.

---

## Notes

- TGFP v2 treats the accent transformation contract itself as something that must be validated
- "Accent" in this context means primarily segmental pronunciation plus bounded lexical-stress/timing changes
- Prosody and emotion preservation are secondary gates, not primary
- The protocol deliberately keeps humans in the loop through all early gates
