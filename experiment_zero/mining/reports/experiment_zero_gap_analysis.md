# Experiment Zero Gap Analysis

## Executive Summary

**Critical finding: No valid same-speaker Indian-English → US-neutral pairs exist in the local corpus.**

The SSD contains 421,050 audio files (214.49 GB) across multiple datasets, but none of them provide the gold-standard supervision AccentEdge needs for Experiment Zero.

## Local Corpus Statistics

| Dataset | Audio Files | English Utterances | With Transcripts | Commercial |
|---|---|---|---|---|
| CMU ARCTIC | 18 speakers | 15,583 | 0 | unknown |
| FLEURS | 4 shards | 2,602 | 0 | yes (CC-BY-4.0) |
| IndicVoices Hindi | 102 shards | 0 (Hindi only) | 450,690 | yes |
| LibriTTS | 33,232 | 33,232 | 0 | yes (CC-BY-4.0) |
| LibriTTS-R 360 | 116,462 | 116,462 | 0 | yes (CC-BY-4.0) |
| Vaani Bangalore | 69 shards | 12,397 | 1,512 | yes (CC-BY-4.0) |
| Vaani Hyderabad | 36 shards | 5,157 | 647 | yes (CC-BY-4.0) |
| VCTK | 43,625 | 43,625 | 0 | yes (CC-BY-4.0) |
| L2-ARCTIC | - | - | - | **RESEARCH ONLY** (CC-BY-NC-4.0) |

## Experiment Zero Matching Results

### Frozen 30 Sentences
- **Exact matches: 0/30**
- **Fuzzy matches: 0/30**
- **Same-speaker repeated sentences: 0**

### Repeated Transcript Search
- English utterances with transcripts: 2,159
- Transcripts appearing 2+ times: 9
- None match Experiment Zero sentences
- Most repeats are Vaani image-description noise templates

## Pair Discovery Results

### Gold Pairs (same verified speaker + exact sentence + dual accent)
**0 found**

### Silver Pairs (same verified speaker + exact sentence + distinct pronunciation)
**0 found**

### Manual Candidates (same probable speaker + same transcript)
**0 found**

### Reference-Only (different speakers, same sentence)
**0 found for Experiment Zero sentences**
9 Vaani internal repeats (same speaker, same descriptive template, different images — not useful for accent conversion)

## What the Local Corpus IS Good For

### AccentEdge-SOURCE (Indian-English distribution)
- **Vaani Bangalore**: 12,397 utterances, 58 speakers, 208.69h
- **Vaani Hyderabad**: 5,157 utterances, 26 speakers, 190.39h
- **Total**: 17,554 utterances, 84 speakers, 399.08h
- **Role**: Source distribution for generalization
- **Limitation**: Image-description speech, not conversational BPO speech

### AccentEdge-REFERENCE-US
- **LibriTTS**: 33,232 utterances
- **LibriTTS-R 360**: 116,462 utterances
- **VCTK**: 43,625 utterances, 108 speakers
- **FLEURS**: 2,602 utterances
- **Total**: ~196K utterances
- **Role**: Target pronunciation reference, rhythm reference, naturalness reference
- **Limitation**: Not same-speaker targets

### Research-Only
- **L2-ARCTIC**: CC-BY-NC-4.0, contains Hindi-background speakers
- **Role**: Pronunciation diagnostics, accent analysis baselines
- **Limitation**: Cannot be used for commercial training

## Critical Gap Analysis

### Gap 1: No Same-Speaker Dual-Accent Pairs
**Severity: BLOCKING**
Local corpus has zero same-speaker pairs where one person performed the same sentence in both Indian and US accents.

**This gap CANNOT be solved by downloading more generic speech data.**
Generic corpora (Common Voice, LibriTTS, etc.) provide accent distributions, not paired transformations.

### Gap 2: No Conversational BPO-Style Speech
**Severity: HIGH**
Vaani data is image-description speech. It does not contain:
- Customer service dialogue
- BPO vocabulary
- Conversational turn-taking
- Phone-call acoustic characteristics

### Gap 3: Limited English Transcripts in Reference Data
**Severity: MEDIUM**
CMU ARCTIC, LibriTTS, and VCTK manifests don't include transcripts in the local manifest. Raw audio exists but transcripts need to be sourced or generated.

### Gap 4: No Experiment Zero Sentence Coverage
**Severity: HIGH**
None of the 30 frozen diagnostic sentences appear in the local corpus. This means:
- No pre-recorded source clips
- No pre-recorded target clips
- Fresh recording is required for all 60 clips

## External Data Strategy

### What External Data CAN Help
1. **Vaani full corpus** (gated, CC-BY-4.0) — more Indian English diversity
2. **Common Voice Indian Accent** (~164h, CC-BY-4.0) — more source speakers
3. **LibriTTS-R full** — more US reference (but we already have 150K+ utterances)

### What External Data CANNOT Help
1. **Same-speaker dual-accent pairs** — no public dataset provides this
2. **BPO conversational speech** — not available in standard corpora
3. **Frozen-30 sentence matches** — these are custom diagnostics

## Recommendation

### Immediate Action (This Week)
1. **Find a dual-accent speaker** — the ONLY way to get gold pairs
2. **Record 60 clips** (30 IN + 30 US) for Experiment Zero
3. **Build A0** and overfit on the 30 pairs

### Secondary Action (After A0)
1. Expand Vaani/Bangalore for Indian English source diversity
2. Use LibriTTS-R + VCTK for US reference distribution
3. Consider L2-ARCTIC for accent diagnostics (research only)

### Do NOT Download
- Do NOT download 100GB+ of LibriTTS-R "just because it exists"
- Do NOT download Common Voice Indian to solve the pair gap (it won't)
- Do NOT download Vaani full corpus before A0 proves the architecture

## Conclusion

The local corpus is **excellent for reference data** (US pronunciation, Indian English source diversity) but **completely empty of same-speaker dual-accent pairs**.

This is expected and normal. AccentEdge's core supervision requirement (same speaker, same sentence, two accents) is rare by design — it's the exact thing we need to create through controlled recording.

**The next action is recording, not downloading.**
