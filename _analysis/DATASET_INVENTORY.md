# AccentEdge — Complete Dataset Inventory

> **Purpose:** Every dataset AccentEdge will need, listed with size, purpose, and bucket
> **Date:** 2026-08-27

---

## Dataset Catalogue (from dossier Section 14 + code references)

### Bucket 1: Production-Safe (after provenance audit)
These are the datasets that can be used for training and target generation once licenses are verified.

| # | Dataset | Size | Speakers | Hours | Language | Role | License | Source |
|---|---------|------|----------|-------|----------|------|---------|--------|
| 1 | **CMU ARCTIC** | ~1.2 GB | 5 (3 US, 1 Indian, 1 other) | ~10 hrs | US English + Indian | Target US / source Indian | Permissive | [festvox.org](http://festvox.org/cmu_arctic/) |
| 2 | **LibriTTS-R** | ~86 GB | ~2,000 | 585 hrs | US English | Target US (primary) | OpenSLR | [openslr.org/141](https://www.openslr.org/141/) |
| 3 | **VCTK** | ~10 GB | 109 | ~44 hrs | Multi-accent English | Speaker identity reference | Edinburgh | [ed.ac.uk](https://datashare.ed.ac.uk/collections/8f1b06bc-ec26-4b8d-ac4e-acb14537d811) |
| 4 | **IndicVoices** | ~100 GB+ | 51,000 | 23,700 hrs | 22 Indian languages | Indian pretraining | CC BY 4.0 | [HF](https://huggingface.co/datasets/ai4bharat/IndicVoices) |
| 5 | **Project Vaani** | ~5000 hrs | ~10,000+ | 5,000 hrs | 12+ Indian languages | Indian pretraining | Open | [vaani.iisc.ac.in](https://vaani.iisc.ac.in/dataset) |
| 6 | **Common Voice Indian** | Variable | Thousands | ~1000+ hrs | Indian English | Source Indian | CC | [HF](https://huggingface.co/datasets/ishands/commonvoice-indian_accent) |
| 7 | **Kaggle Indian English** | Variable | ~200 | ~50 hrs | Indian English | Source Indian | Kaggle | [kaggle](https://www.kaggle.com/datasets/kotekalvijay/indian-english-accent-audio) |

### Bucket 2: Research / Evaluation Only
Non-commercial or restrictive licenses — usable for experiments and benchmarks but NOT for training production models.

| # | Dataset | Size | Speakers | Hours | Language | Role | License | Source |
|---|---------|------|----------|-------|----------|------|---------|--------|
| 8 | **L2-ARCTIC** | ~4 GB | 24 (6 L1s including Hindi) | ~29 hrs | Non-native English | Source Indian (research) | Non-commercial | [psi.tamu.edu](https://psi.engr.tamu.edu/l2-arctic-corpus/) |
| 9 | **Speech Accent Archive** | ~2 GB | ~2,000 | ~100 hrs | 100+ accents | Accent diversity | Research | [accent.gmu.edu](https://accent.gmu.edu/) |
| 10 | **FLEURS** | ~5 GB | Many | ~100+ hrs | 102 languages (en_in, en_us) | Evaluation | CC BY 4.0 | [HF](https://huggingface.co/datasets/google/fleurs) |

### Bucket 3: License Clarification Needed
Downloadable but commercial model-training rights unclear.

| # | Dataset | Size | Speakers | Hours | Language | Role | Concern | Source |
|---|---------|------|----------|-------|----------|------|---------|--------|
| 11 | **VCTK (via HF mirror)** | ~10 GB | 109 | ~44 hrs | Multi-accent | Speaker identity | Verify original Edinburgh terms | [HF](https://huggingface.co/datasets/vt57299/vctk) |

### Bucket 4: Proprietary (to be collected)
AccentEdge's own paired data — the moat.

| # | Dataset | Size | Speakers | Hours | Language | Role | Status | Source |
|---|---------|------|----------|-------|----------|------|--------|--------|
| 12 | **AccentEdge-PAIR** | TBD | 50+ (target) | ~200 hrs (target) | Indian English → US-neutral | Paired same-speaker target | NOT YET COLLECTED | Self-collected |

---

## Dataset Use by Phase

| Phase | Datasets Used | Purpose |
|-------|---------------|---------|
| **Phase 0** | CMU ARCTIC (1 speaker), L2-ARCTIC | TGFP v2 target generation, annotation |
| **Phase 1** | CMU ARCTIC, L2-ARCTIC, LibriTTS-R | Benchmark instrument, evaluation |
| **Phase 2** | CMU ARCTIC, LibriTTS-R, VCTK | Architecture bake-off (candidates A/B/C/D) |
| **Phase 3** | IndicVoices, Project Vaani, Common Voice Indian, CMU ARCTIC, LibriTTS-R | Training data (source + target) |
| **Phase 7+** | AccentEdge-PAIR (proprietary) | Fine-tuning, production |

---

## Key Observations

1. **No Indian-accented paired data exists openly.** Open datasets give you Indian speech AND American speech separately, but never the same person saying the same sentence in both accents. That's why AccentEdge-PAIR must be collected.

2. **L2-ARCTIC is critical but non-commercial.** It's the only open dataset with Indian-English speakers (Hindi L1) reading the SAME sentences as native speakers. This makes it essential for Phase 0/1 research but blocks production training.

3. **IndicVoices is huge (23.7K hours) but mostly Indic languages.** Only a subset is English. It's better for pretraining an Indian-accent encoder than for accent conversion itself.

4. **LibriTTS-R (86GB) is the primary US target corpus.** It's the largest, cleanest, permissively-licensed US English speech corpus available.

5. **CMU ARCTIC is tiny (~10 hrs) but unique.** It has Indian-accented speakers (rks — "Indian male") reading the SAME 1,150 sentences as US speakers. This makes it the only open same-sentence cross-accent dataset.

6. **Total download size if cloning everything:** ~115–120 GB
   - Bucket 1: ~97 GB (LibriTTS-R dominates)
   - Bucket 2: ~11 GB
   - Bucket 3: ~10 GB (VCTK)

7. **Disk needed after extraction:** ~200 GB (audio uncompresses 3–10×)

---

## Recommended Clone Order

| Priority | Dataset | Size | Why First |
|----------|---------|------|-----------|
| 1 | CMU ARCTIC | 1.2 GB | Phase 0 needs it NOW — same sentences across accents |
| 2 | L2-ARCTIC | 4 GB | Phase 0 needs Indian English source |
| 3 | LibriTTS-R | 86 GB | Phase 1+ target corpus, largest by far |
| 4 | VCTK | 10 GB | Speaker identity evaluation |
| 5 | IndicVoices | 100 GB+ | Phase 3 Indian pretraining (large) |
| 6 | Project Vaani | ~50 GB | Phase 3 Indian pretraining |
| 7 | FLEURS | 5 GB | Phase 1 evaluation |
| 8 | Common Voice Indian | ~20 GB | Phase 3 source diversity |
| 9 | Kaggle Indian English | ~5 GB | Phase 3 source supplement |

---

## What Needs License Review Before Training

- **L2-ARCTIC**: Non-commercial only — blocks production training
- **VCTK**: Verify Edinburgh license terms (commercial use unclear)
- **FAcodec** (codec model, not dataset): License must be verified — entire codec path depends on it
