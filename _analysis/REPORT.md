# AccentEdge — Complete Codebase Analysis

> **Date:** 2026-08-27  
> **Scope:** Full monorepo (~164 source files, 42 test files)  
> **Purpose:** Explain what this codebase does, how it works, and where it stands

---

## 1. Executive Summary

AccentEdge is a **research project, not a product**. Its goal is to answer one question: *can we transform Indian-English speech into US-neutral pronunciation while keeping the same speaker, same words, same emotion?* If yes, the technology goes into BPO call centres to improve agents' accent intelligibility without voice cloning.

The codebase spans **9 research phases** (0–8), of which only Phase 2 is complete. The monorepo contains everything: experiment harnesses, neural model code, 5 streaming architecture candidates, a full benchmark suite, and training infrastructure. It is functional but has 8 broken imports and 4 missing dependencies that must be fixed before it can import cleanly.

---

## 2. What AccentEdge Actually Does (Non-Technical)

Think of it as a **pronunciation coach that runs inside a neural network**. Given a recording of an Indian-English speaker saying "I can see a charge of thirty dollars", it produces a new recording of the same person saying the exact same words, but with American pronunciation patterns (flapped /t/, raised /æ/, etc.).

The critical constraint: **it must not sound like a different person**. This is harder than it looks — most voice conversion systems change the speaker's identity along with the accent. AccentEdge's entire research program is about proving this identity preservation is achievable.

---

## 3. Architecture Overview

```
accentedge/
├── pyproject.toml          ← Single dependency lock (unified from 4 projects)
├── README.md               ← Project overview + phase roadmap
├── docs/                   ← All specification and audit documents
│   ├── phase0/             ← TGFP v2, feasibility spec, forensic audit
│   └── phase1/             ← Implementation status, architecture decisions, audit reports
│
├── FAcodec/                ← Symlink to ~/FAcodec (neural speech codec, external)
├── Amphion/                ← (future) Git submodule for TTS toolkit
│
├── src/accentedge/
│   ├── phase0/             ← 17 modules — experiment harness, target generation, evaluation
│   ├── phase1/             ← 6 modules — FAC-FACodec model (denoiser + diffusion)
│   ├── benchmark/          ← 47 modules — BPO evaluation instrument
│   ├── codec/              ← 2 modules — FACodec adapter interface
│   ├── evaluation/         ← 4 modules — acoustic, identity, content, phoneme metrics
│   ├── models/             ← 34 modules — 5 streaming candidates + interfaces
│   ├── streaming/          ← 7 modules — virtual-time simulator, chunker, timeline
│   ├── training/           ← 8 modules — dataset, checkpoint, loss, trainer
│   ├── audio/              ← 4 modules — low-level I/O, VAD, buffer, playback
│   ├── config/             ← 2 modules — YAML schema + loader
│   ├── data/               ← 3 modules — lineage, validation, schemas
│   ├── experiments/        ← 1 module — experiment registry
│   ├── profiling/          ← 4 modules — latency, RTF, memory profiling
│   ├── reporting/          ← 3 modules — HTML/JSON reports, ADR generation
│   ├── cli/                ← CLI entry point
│   ├── metrics/            ← (empty)
│   └── utils/              ← (empty)
│
├── tests/                  ← 42 test files (flat layout, no subpackage mirroring)
├── checkpoints/            ← Model weights, evaluation results, WAV files
├── data/                   ← Samples, gold targets, manifests, alignment data
├── configs/                ← YAML configs (phase0, phase1, benchmark, splits)
└── scripts/                ← 25 entry-point scripts (colab, training, gates, etc.)
```

### How the Phases Connect

```
Phase 0: Target Feasibility
    │  Can we construct a usable accent-transformation target?
    │  Step 0: 1 speaker, 1 sentence, Strategy B, hand-built, listen
    │  Gates: Gate 1 (reconstruction quality), Gate 2 (identity preservation)
    │
    ▼ passes
Phase 1: BPO Benchmark
    │  Build a speaker-disjoint, leakage-resistant evaluation instrument
    │  Output: benchmark/ suite with manifests, annotations, runners
    │
    ▼
Phase 2: Architecture Bake-off  ← DONE
    │  Compare 5 streaming S2S architectures (A/B/C/D + Sparse Repair)
    │  Winner: Candidate D (Minimal Hybrid)
    │
    ▼
Phase 3: AccentEdge S2S Model
    │  First proprietary model — train the actual accent converter
    │  Uses Phase 2's chosen architecture
    │
    ▼
Phase 4: Streaming Inference
    │  Real-time chunked conversion (150 ms target latency)
    │
    ▼
Phase 5: Optimisation
    │  Latency, memory, quality tuning
    │
    ▼
Phase 6: Runtime
    │  Windows endpoint, admin portal
    │
    ▼
Phase 7: Pilot
    │  BPO deployment pilot
    │
    ▼
Phase 8: Production
       Full production rollout
```

---

## 4. Deep Dive by Phase

### Phase 0 — Target Feasibility (17 modules, in progress)

**Thesis:** Before building any model, prove the target is constructible.

**Experiment Flow:**
1. **Load source audio** (Indian-English speaker) + gold (US-neutral target)
2. **Annotate** both with phone-level labels and forced-alignment timestamps
3. **Generate target** using one of three strategies (A: parametrised, B: patchwork, C: TTS)
4. **Degrade** both source and target with codec/bandwidth/phone artifacts
5. **Evaluate** across four gates:
   - Gate 1: Reconstruction quality (source → codec → source, must preserve audio)
   - Gate 2: Identity preservation (source vs. target must sound like the same person)
   - Gate 3: Content preservation (transcription must match exactly)
   - Gate 4: Accent conversion strength (target must be measurably more US-neutral)
6. **Listen** — human ABX/listener study with realization labels
7. **Report** — per-gate outcome + provenance record

**Key modules:**
- `experiment.py` — top-level controller; registers sources, golds, targets; runs gates
- `annotations.py` — phone-level annotation DB (TIMIT-style phones, tokens, alignments)
- `target_generation.py` — Strategy A/B/C: parametrised morphing, patchwork splicing, TTS
- `degradation.py` — codec compression, bandwidth limiting, packet loss simulation
- `evaluation.py` — four-gate evaluation logic with pass/fail thresholds
- `listening.py` — listener study framework (ABX, MOS, preference tests)
- `realization_labels.py` — DEVIANT / ALREADY-TARGET / AMBIGUOUS taxonomy

**Status:** Code is functional but gates have not been run yet. No WER or identity results exist.

---

### Phase 1 — FAC-FACodec Model (6 modules, scaffolded)

**Thesis:** If Phase 0 proves the target is constructible, Phase 1 builds the model that approximates it.

**The Model (from paper `arxiv:2510.10785v2`):**
- Takes FACodec's factorized latents as input
- Factorizes the latent space: content codebooks (2) + prosody codebook (1) + residual codebooks (3) + timbre
- Modifies only the content codebooks (where accent lives)
- Uses a small denoiser + diffusion model to refine the modified latents
- Decodes back to waveform through FACodec's decoder

**Key modules:**
- `codec/interfaces.py` — `FactorizedLatents` dataclass + `FactorizedSpeechCodec` protocol
- `codec/facodec.py` — `FACodecAdapter` that wraps ~/FAcodec/ for encode/decode
- `phase1/diffusion.py` — small DDPM denoiser for latent refinement
- `phase1/denoiser.py` — lightweight denoising network
- `phase1/converter.py` — `AccentConverter`: the main model (encode → modify → refine → decode)
- `phase1/strength.py` — conversion strength control (how much accent change)
- `phase1/phoneme_pipeline.py` — transcript → frame-level phoneme IDs aligned to codec FPS

**Critical finding from analysis:** The code's frame rate (codec_hop=300 at 24kHz = 80 fps) does NOT match the FAC-FACodec paper's reported 50 fps. This is a **potential correctness issue** — all training data alignment assumes 80 fps.

**Status:** Scaffolded. No training driver exists (identified as a gap). No checkpoints beyond `final_model.pt` (appears to be a test artifact).

---

### Phase 2 — Architecture Bake-off (34 model modules, COMPLETE)

**Thesis:** Before committing to one architecture, compare 5 streaming S2S candidates and pick the winner.

**The Five Candidates:**

| Candidate | Approach | Lookahead | Latency | Complexity |
|-----------|----------|-----------|---------|------------|
| **A** — Streaming AC | Encoder → accent bottleneck → HiFi-GAN synth | 640 ms (paper) / 0 ms (low) | 30–640 ms | Medium |
| **B** — Articulatory DDSP | Encoder → articulatory params → DDSP synth | 0 ms | ~30 ms | Medium |
| **C** — Token Translation | Tokenise content → translate tokens → synthesise | 0 ms | ~40 ms | Medium-High |
| **D** — Minimal Hybrid | Codec swap + 2-layer mapper | **0 ms** | **~10 ms** | **Low (selected)** |
| **Sparse Repair** | Post-hoc repair of sparse latent errors | 0 ms | ~10 ms | Low |

**Winner: Candidate D (Minimal Hybrid)**
- Encode with FACodec → swap content codebook → decode
- 2-layer linear mapper for accent transformation
- Zero lookahead, ~10 ms algorithmic latency
- Simplest architecture, lowest compute

**Key finding:** Most candidates are **stub implementations** (22 stubs identified). Only Candidate D has a complete model file. The others define interfaces but most methods are `pass` or return placeholder tensors.

**Key modules:**
- `models/interfaces.py` — `StreamingCandidate` protocol + `StreamingSession` + `StreamingResult`
- `models/registry.py` — `ModelRegistry`: register, get, list, instantiate candidates
- `streaming/simulator.py` — virtual-time simulator with backlog/RTF tracking
- `streaming/chunker.py` — audio chunking with configurable window sizes
- `streaming/timeline.py` — event timeline for latency analysis
- `streaming/session.py` — streaming session state management

---

### Benchmark (47 modules, scaffolded)

**Thesis:** Evaluation instrument for comparing accent conversion systems.

**Architecture:**
- `runner/benchmark.py` — `BenchmarkRunner`: processes full utterances through a `CandidateAdapter`
- `runner/run_manifest.py` — creates run records with git-style lineage
- `runner/failures.py` — error classification (`ErrorCategory` enum)
- `candidates/` — adapter pattern: `PassthroughAdapter`, `FileOutputAdapter`, `RegistryAdapter`
- `schemas/` — `DatasetItem`, `RunManifest`, `BenchmarkResult` (Pydantic models)
- `statistics/` — aggregation, bootstrap CI, paired statistical tests
- `evaluation/` — content WER, pronunciation, prosody, naturalness, timing, robustness
- `identity/` — speaker similarity, calibration curves
- `alignment/` — forced alignment, TextGrid I/O, validation
- `degradations/` — codec/bandwidth degradation for robustness testing
- `reporting/` — HTML reports with tables, JSON output

**Key finding:** The benchmark has **zero tests** (47 files, 0 test coverage). This is the highest-risk gap.

---

### Training & Data Pipeline (8 modules, partial)

**Thesis:** Load audio, extract FACodec latents, align phonemes, yield training batches.

**Pipeline:**
1. `torchaudio.load()` → mono float32 waveform
2. `FACodecAdapter.encode()` → `FactorizedLatents` (content_zc1 codebook indices)
3. `PhonemePipeline` → transcript → phoneme IDs aligned to codec FPS (80 fps)
4. Cache to disk (`latents.pt`, `phone_ids.pt`, `metadata.json`)
5. `collate_fn` pads variable-length sequences → `[B, 1, T_max]` batch
6. `NativePriorDataset` yields `{zc1, phone_ids, valid_mask, speaker_id, item_id}`

**Training modules:**
- `training/losses.py` — combined loss (cross-entropy + CTC + optional diffusion)
- `training/trainer.py` — main training loop with checkpointing
- `training/checkpoints.py` — save/load with manifest
- `training/schedules.py` — LR schedules + optimiser
- `training/overfit.py` — overfit-on-batch sanity test
- `training/reproducibility.py` — seed setting, deterministic flags

**Critical finding:** There is **no training driver script**. The `trainer.py` module exists but nothing calls it. The `scripts/train_phase1.py` exists but has not been verified to work end-to-end.

---

### Audio & Codec Infrastructure (6 modules)

- `audio/` — low-level primitives: `buffer.py` (ring buffer), `capture.py` (mic input), `playback.py` (audio output), `vad.py` (voice activity detection)
- `codec/` — `FACodecAdapter` wraps ~/FAcodec/ for encode/decode; `interfaces.py` defines the protocol

---

### Config, Data, Experiments, Reporting (9 modules)

- `config/` — YAML schema (`schema.py`) + loader (`loader.py`)
- `data/` — lineage tracking, schema validation
- `experiments/` — experiment record + registry (from P2)
- `profiling/` — latency, RTF, memory profiling
- `reporting/` — HTML report generation, ADR (Architecture Decision Record) generation

---

## 5. External Dependencies

| Dependency | Used By | Why |
|------------|---------|-----|
| `torch >=2.2` | All model code | Neural network framework |
| `torchaudio >=2.2` | audio/, training/ | Audio I/O + resampling |
| `numpy >=1.24` | Everything | Array operations |
| `scipy >=1.11` | audio, evaluation | Signal processing |
| `soundfile >=0.12` | audio I/O | WAV file reading |
| `librosa >=0.10` | evaluation/acoustic | Mel spectrograms |
| `speechbrain >=0.5.16` | evaluation/identity | ECAPA-TDNN speaker embeddings |
| `faster-whisper >=1.0` | phase1/phoneme_pipeline | Whisper ASR for content evaluation |
| `phonemizer >=3.0` | phase1/phoneme_pipeline | G2P conversion (text → phones) |
| `jiwer >=3.0` | evaluation, benchmark | WER calculation |
| `pydantic >=2.0` | benchmark, config | Data validation |
| `pandas >=2.0` | benchmark, statistics | DataFrame analysis |
| `pyarrow >=12.0` | benchmark | Parquet I/O for manifests |
| `typer >=0.9` | benchmark CLI | CLI framework |
| `jinja2 >=3.1` | reporting | HTML template rendering |
| `matplotlib >=3.7` | reporting | Plot generation |
| `pyyaml >=6.0` | config, experiments | YAML config parsing |
| `huggingface-hub >=0.20` | model loading | HF model download |
| `einops >=0.7` | model code | Tensor rearrangement |

---

## 6. Code Quality Assessment

### What's Solid
- **Monorepo structure is clean** — single pyproject.toml, unified src layout, all imports resolved
- **No technical debt markers** — zero TODO/FIXME/HACK/XXX comments
- **No syntax errors** — all 164 files parse cleanly
- **Interfaces are well-defined** — `FactorizedSpeechCodec` and `StreamingCandidate` are proper protocols with clear contracts
- **Caching layer** — FACodec latent extraction + phoneme alignment are cached to disk, avoiding recomputation
- **Benchmark has good foundations** — Pydantic schemas, error classification, run manifests, statistical rigour

### What's Fragile
| Issue | Severity | Details |
|-------|----------|---------|
| **8 broken internal imports** | Critical | `accentedge.models.models.interfaces` (nested `models/` path), `accentedge.codec.facodec` line 82 (wrong import), 6 more — package fails on import |
| **4 missing pyproject.toml deps** | Critical | `typer`, `jinja2`, `matplotlib`, `torchaudio` — import will fail even if torch is installed |
| **Frame rate mismatch** | High | Codec hop=300 @ 24kHz = 80 fps, but FAC-FACodec paper reports 50 fps. All phoneme alignment and training data assumes 80 fps. |
| **22 stub implementations** | Medium | Most Phase 2 candidate methods are `pass` or return zeros. Only Candidate D is complete. |
| **0 tests for benchmark/** | High | 47 benchmark files, zero test coverage |
| **0 tests for models/** | High | 34 model files, zero test coverage |
| **No training driver** | High | `training/trainer.py` exists but nothing invokes it. `scripts/train_phase1.py` is unverified. |
| **Empty subpackages** | Low | `metrics/` and `utils/` are empty directories |
| **Flat test layout** | Low | Tests are flat under `tests/` instead of mirroring `src/` structure |

### What's Missing
1. **End-to-end training pipeline** — from raw audio to trained model
2. **Tests for the two largest subpackages** — benchmark (47 files) and models (34 files)
3. **Real data** — no actual Indian-English speech dataset is included
4. **Amphion integration** — git submodule not set up
5. **Phase 0 execution** — no gates have been run, no listener study data
6. **Documentation for developers** — README explains the project but not how to extend it

---

## 7. Immediate Next Steps (Ranked by Priority)

| # | Action | Effort | Impact |
|---|--------|--------|--------|
| 1 | **Fix 8 broken imports** | 5 min | Unblocks entire package |
| 2 | **Add 4 missing deps to pyproject.toml** | 5 min | Unblocks benchmark, reporting |
| 3 | **Investigate frame rate mismatch** | 30 min | Could invalidate all training data |
| 4 | **Write training driver script** | 2–3 hrs | Unblocks Phase 1 model training |
| 5 | **Add tests for benchmark/** | 1–2 hrs | Makes 47-file module safe to modify |
| 6 | **Add tests for models/** | 2–3 hrs | Makes 34-file module safe to modify |
| 7 | **Set up Amphion submodule** | 15 min | Enables Colab workflows |
| 8 | **Mirror test directories** | 1 hr | Makes coverage tracking automatic |
| 9 | **Fill metrics/ and utils/** | TBD | Remove empty packages |
| 10 | **Run Phase 0 gates** | Weeks | Actual research milestone |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Frame rate mismatch invalidates training data | Medium | **Critical** | Verify against FACodec source, regenerate cache if wrong |
| Broken imports block any import | **Certain** | High | Fix #1 above (5 min) |
| Phase 2 candidates are mostly stubs | **Certain** | Medium | Only Candidate D is meant to be complete; others are reference designs |
| No training driver means Phase 1 is non-functional | **Certain** | High | Write driver script (#4 above) |
| Zero benchmark test coverage | **Certain** | Medium | Add tests (#5 above) |
| Phase 0 gates may fail (research risk) | Unknown | **Critical** | That's what Phase 0 is for — it's designed to fail fast |
| Missing real speech data | **Certain** | **Critical** | No dataset is included; must acquire or generate |
| FACodec dependency on external repo | Low | Medium | Symlink to ~/FAcodec/ is fragile; git submodule preferred |

---

## 9. Summary

**AccentEdge is a well-structured research codebase with a clear thesis and rigorous phase-gated methodology.** The monorepo consolidation merged 4 projects cleanly (164 source files, 42 tests, 9 phases). The architecture is sound: FACodec-based factorized latents, a clear separation between model and benchmark, and a streaming simulator for latency analysis.

**The two immediate blockers are:** (1) 8 broken imports prevent any module from loading, and (2) 4 missing dependencies prevent the benchmark and reporting modules from running. Both are 5-minute fixes.

**The two strategic risks are:** (1) a potential frame-rate mismatch between the code and the FAC-FACodec paper that could invalidate all training data, and (2) the complete absence of a training driver, meaning Phase 1 is currently non-functional despite having model code.

The project is research-stage, not production. Its purpose is to validate a thesis, not to ship a product. The code quality is clean (no TODOs, no syntax errors, well-defined interfaces), but the testing coverage is thin for the two largest subpackages.

---

*Analysis generated from 8 parallel codebase audits, verified by independent reviewer.*  
*Files: `_analysis/01_architecture.md`, `02_phase0.md`, `03_phase1_model.md`, `04_phase2_models.md`, `05_benchmark.md`, `06_training_data.md`, `07_quality_audit.md`, `08_verification.md`*
