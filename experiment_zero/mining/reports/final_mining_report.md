# AccentEdge — Local Speech Corpus Mining Report

## Executive Summary

**No valid same-speaker Indian-English → US-neutral pairs exist in the local corpus.**

The SSD contains 421,050 audio files (214.49 GB) across 9 datasets, but none provide the gold-standard supervision AccentEdge needs for Experiment Zero. The frozen 30-sentence diagnostic matrix has zero exact matches in the corpus. Zero same-speaker repeated transcripts were found.

**Conclusion: We must record 60 fresh clips (30 IN + 30 US) with a controlled dual-accent speaker.**

---

## Local Corpus Summary

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

**Total**: 421,050 audio files, 214.49 GB, 229,058 English utterances, 2,159 with transcripts

---

## Experiment Zero Matching Results

### Frozen 30 Sentences
- **Exact matches: 0/30**
- **Fuzzy matches: 0/30**
- **Same-speaker repeated sentences: 0**

### Repeated Transcript Search
- English utterances with transcripts: 2,159
- Transcripts appearing 2+ times: 9
- None match Experiment Zero sentences
- Most repeats are Vaani image-description noise templates (same speaker describing different images)

---

## Pair Discovery Results

### Gold Pairs (same verified speaker + exact sentence + dual accent)
**0 found**

### Silver Pairs (same verified speaker + exact sentence + distinct pronunciation)
**0 found**

### Manual Candidates (same probable speaker + same transcript)
**0 found**

### Reference-Only (different speakers, same sentence)
**0 found for Experiment Zero sentences**

---

## What We Have

### AccentEdge-SOURCE (Indian English)
- **Vaani Bangalore**: 12,397 English utterances, 58 speakers, 208.69h
- **Vaani Hyderabad**: 5,157 English utterances, 26 speakers, 190.39h
- **Total**: 17,554 English utterances, 84 speakers, ~20.62h English
- **IndicVoices Hindi**: 450,690 Hindi utterances (acoustic pretraining, not Indian-English)
- **Common Voice Indian Accent**: 110k clips, ~164h (HF, commercial OK)

### AccentEdge-REFERENCE-US
- **LibriTTS**: 33,232 utterances, 247 speakers, ~70h
- **LibriTTS-R 360**: 116,462 utterances, 904 speakers, ~360h
- **VCTK**: 43,625 utterances, 108 speakers, ~48h
- **CMU ARCTIC**: 15,583 utterances, 18 speakers, ~15h
- **FLEURS**: 2,602 utterances (Indian English subset, US-oriented)

### AccentEdge-PAIR (same-speaker dual-accent)
**NONE FOUND**

---

## External Gaps

**What we still need:**
1. **30 same-speaker paired utterances** (Indian-English → US-neutral) — NOT available locally
2. **No same-speaker dual-accent dataset identified** on Hugging Face or Kaggle that we can verify

**External candidates to investigate:**
- **Common Voice Indian Accent** (ishands/commonvoice-indian_accent): ~110k clips, ~164h, Indian-accent English. Useful for source distribution, NOT for pairs.
- **Vaani** (ARTPARK-IISc/Vaani): CC-BY-4.0, extremely large, gated. Useful for Indian-English diversity.
- **L2-ARCTIC**: CC-BY-NC-4.0, contains Hindi-background speakers. RESEARCH ONLY.
- **Common Voice en-IN**: Empty locally.

---

## Recommendation

**Do not download more data.**

The gap is not dataset size. The gap is **supervision structure**:

> We need 30 pairs where the same speaker produced both the Indian-English and US-neutral versions of the same sentence.

This requires **controlled recording**, not more downloads.

### Next Actions
1. Find one bilingual speaker who can reliably perform both accents
2. Record 60 clips (30 IN + 30 US) using the frozen diagnostic matrix
3. Manually listen to every pair
4. Reject bad pairs
5. Align/normalize
6. Build A0 overfit

---

## Licensing Summary

| License | Datasets | Hours | Commercial Use |
|---|---|---|---|
| CC-BY-4.0 | Vaani, LibriTTS, VCTK, FLEURS | ~700h | YES |
| CC-BY-NC-4.0 | L2-ARCTIC | ~? | RESEARCH ONLY |
| Unknown | CMU ARCTIC, Speech Accent Archive | ~30h | UNVERIFIED |
| Hindi-only | IndicVoices | 828h | YES (not Indian-English) |

---

## Files Generated

- `inventory/files_audio.parquet` — 421,050 audio file records
- `inventory/files_metadata.parquet` — 805,168 metadata file records
- `inventory/files_archives.parquet` — 164 archive records
- `manifests/datasets.parquet` — 16 dataset records with licensing
- `candidates/experiment_zero_text_matches.parquet` — 0 matches
- `candidates/repeated_transcripts.parquet` — 2,030 unique English transcripts
- `reports/experiment_zero_gap_analysis.md` — detailed gap analysis
- `reports/final_mining_report.md` — this file

---

**Mining complete. No external downloads warranted until recording happens.**
