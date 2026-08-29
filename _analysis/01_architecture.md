# AccentEdge Monorepo Architecture Analysis

## 1. Project Thesis

**AccentEdge is an identity-preserving speech-to-speech accent conversion system that transforms Indian English to US-neutral pronunciation for BPO call-center environments, investigated through a sequenced, gate-controlled research protocol running across nine phases in a single Python monorepo.**

---

## 2. How the 9 Phases Connect

The nine phases form a strictly sequential, gate-gated pipeline where each phase must satisfy exit criteria before the next begins:

```
Phase 0: Target Feasibility
    ↓ constructs usable accent-transformation targets and validates identity preservation
Phase 1: BPO Benchmark
    ↓ builds a speaker-disjoint, leakage-resistant evaluation instrument
Phase 2: Architecture Bake-off
    ↓ evaluates 5 streaming candidates and selects Candidate D (Minimal Hybrid)
Phase 3: AccentEdge S2S Model
    ↓ trains the first proprietary accent-conversion model
Phase 4: Streaming Inference
    ↓ implements real-time chunked/streaming conversion
Phase 5: Optimisation
    ↓ tunes latency, memory, and quality trade-offs
Phase 6: Runtime
    ↓ packages Windows endpoint and admin portal
Phase 7: Pilot
    ↓ deploys to a BPO pilot site
Phase 8: Production
    ↓ full production rollout
```

**Phase 0 and Phase 1 are the only in-progress / scaffolded phases in the codebase.** Phase 2 is marked complete in documentation (Candidate D selected). Phases 3–8 are planned. The phase modules in `src/accentedge/` are therefore:

- `phase0/` — active research code for target feasibility (TGFP v2).
- `phase1/` — scaffolded code for the BPO benchmark instrumentation.
- `models/` and `streaming/` — contain the Phase 2 bake-off candidates (Candidate A–E), with Candidate D (`minimal_hybrid`) being the selected streaming architecture.

Data flows upward: Phase 0 produces validated targets and gold-standard recordings, which Phase 1 benchmarks, which Phase 2 evaluates across architecture candidates, which then inform the training data and evaluation criteria for Phase 3 onward.

---

## 3. Monorepo Package Structure and Roles

The repository uses a **src-layout** (`src/accentedge/`) with 18 subpackages. Each subpackage has a distinct architectural role:

### Core Phase Packages
| Subpackage | Role |
|---|---|
| `phase0/` | **Target Feasibility.** TGFP v2 experiment framework: experiment controller/runner, gate sequence execution, target generation strategies, degradation pipelines, alignment, transcription, identity-transfer metrics, provenance tracking, annotations, listening tests, and Phase 0 reporting. |
| `phase1/` | **BPO Benchmark (Scaffolded).** FAC-FACodec-based model infrastructure: diffusion model, denoiser, phoneme pipeline, converter, and ZC2 recompute utilities. |

### Model and Inference Packages
| Subpackage | Role |
|---|---|
| `models/` | **Architecture Candidates.** Registry and interfaces for 5 streaming candidates: `minimal_hybrid` (Candidate D, selected), `sparse_repair`, `articulatory_ddsp`, `streaming_ac`, and `token_translation`. Each exposes `interfaces.py` and `streaming_config.py`. |
| `streaming/` | **Streaming Runtime Primitives.** Virtual-time simulator, chunker, session manager, timeline, causality enforcement, continuity checks, and state-growth profiling for low-latency streaming inference. |
| `codec/` | **FACodec Adapter.** Wraps the Plachta FAcodec factorized audio codec behind `interfaces.py`, providing latent representations used by Phase 1 and benchmark pipelines. |

### Evaluation and Benchmarking Packages
| Subpackage | Role |
|---|---|
| `evaluation/` | **Metrics Library.** Acoustic (STOI/PESQ/MCD), content (WER), identity (speaker similarity), and phoneme-level evaluation tools. |
| `benchmark/` | **BPO Benchmark Suite.** Complete evaluation instrument: sweep runners, candidate comparison, adapter integration, schema validation, CLI, reporting, degradation application, alignment, and dataset management. |

### Training and Experimentation Packages
| Subpackage | Role |
|---|---|
| `training/` | **Model Training Toolkit.** Dataset loaders, checkpoint management, overfit tests, trainer loop, loss functions, learning-rate schedules, and reproducibility helpers. |
| `experiments/` | **Experiment Registry.** Lightweight registry for tracking experiment configurations and results across phases. |

### Infrastructure and Support Packages
| Subpackage | Role |
|---|---|
| `config/` | **Configuration Layer.** Pydantic-based schema definitions (`Phase2Config`, `TrainingConfig`, `StreamingConfig`, etc.) and YAML/JSON loader utilities. |
| `data/` | **Data Governance.** Data lineage tracking, schema definitions, and validation logic for audio manifests and metadata. |
| `audio/` | **Audio Primitives.** Buffer management, capture, playback, and VAD wrappers used across Phase 0, benchmark, and streaming. |
| `profiling/` | **Performance Profiling.** Latency, memory, real-time-factor (RTF), and hardware introspection tools. |
| `reporting/` | **Report Generation.** ADR (Architecture Decision Record) generation, Pareto frontier analysis, and Phase 2 HTML/JSON report builders. |
| `metrics/` | **General Metrics.** Supplementary metric utilities (present in directory tree). |
| `utils/` | **General Utilities.** Shared helper functions (present in directory tree). |
| `cli/` | **Command-Line Interface.** Top-level CLI entry point for the package. |

### Top-Level Package
| Subpackage | Role |
|---|---|
| `accentedge/` (`__init__.py`) | **Namespace Root.** Defines `__version__ = "0.1.0-monorepo"` and project docstring. |

---

## 4. External Dependencies and Why Each Is Needed

| Dependency | Version Constraint | Purpose |
|---|---|---|
| `torch` | >= 2.2 | Core deep-learning framework for all model training and inference (Phase 1 models, Candidate D). |
| `numpy` | >= 1.24 | Array and numerical operations for audio waveforms, metrics, and signal processing. |
| `scipy` | >= 1.11 | Signal processing primitives (filters, interpolation) for audio and degradation. |
| `einops` | >= 0.7 | Tensor shape manipulation/rearrangement in model architectures. |
| `soundfile` | >= 0.12 | Audio file I/O (read/write WAV, FLAC) via libsndfile. |
| `librosa` | >= 0.10 | Audio loading, feature extraction (mel spectrograms, STFT), and resampling. |
| `torchaudio` | >= 2.2 | GPU-accelerated audio transforms and I/O aligned with PyTorch. |
| `jiwer` | >= 3.0 | Word Error Rate (WER) computation for content-preservation evaluation. |
| `speechbrain` | >= 0.5.16 | Pretrained ASR and speaker-embedding models for content and identity metrics. |
| `faster-whisper` | >= 1.0 | Optimized Whisper ASR for transcription and WER evaluation. |
| `phonemizer` | >= 3.0 | Grapheme-to-phoneme conversion for phoneme-level accent analysis. |
| `pyyaml` | >= 6.0 | YAML serialization/deserialization for experiment configs and phase configs. |
| `pydantic` | >= 2.0 | Data validation and settings management for configuration schemas. |
| `pandas` | >= 2.0 | Tabular data handling for evaluation results, manifests, and sweep outputs. |
| `pyarrow` | >= 12.0 | Parquet/Arrow columnar serialization for large benchmark datasets. |
| `typer` | >= 0.9.0 | CLI framework for benchmark and experiment command-line tools. |
| `jinja2` | >= 3.1 | HTML report templating for Phase 2 reports and benchmark dashboards. |
| `matplotlib` | >= 3.7 | Plotting for Pareto frontiers, latency histograms, and evaluation visualizations. |
| `huggingface-hub` | >= 0.20 | Downloading and caching pretrained model weights (Whisper, SpeechBrain) from HuggingFace Hub. |

---

## 5. Build / Install Flow

### Build System
- **PEP 517/518** with `setuptools >= 68` and `wheel`.
- Build backend: `setuptools.backends.legacy:build`.
- **src-layout**: `[tool.setuptools.packages.find] where = ["src"]` and `[tool.setuptools.package-dir] "" = "src"`.

### Install Flow
1. **Editable install (development):**
   ```bash
   pip install -e .
   ```
   This installs the `accentedge` package in editable mode, pointing to `src/accentedge/`, with all required dependencies.

2. **Optional extras:**
   - **CUDA:** `pip install -e ".[cuda]"` — pulls `torch[cuda] >= 2.2`.
   - **MPS (Apple Silicon):** `pip install -e ".[mps]"` — pulls `torch >= 2.2` with platform-specific marker.

3. **Testing:**
   - `pytest` with configuration in `[tool.pytest.ini_options]`.
   - Test discovery: `tests/test_*.py`, test classes `Test*`, test functions `test_*`.
   - Dev extras include `pytest`, `pytest-cov`, `pytest-xdist` for parallel execution.

4. **Linting and Typing:**
   - **Ruff** (`[tool.ruff.lint]`): enabled for `E` (pycodestyle errors), `F` (pyflakes), and `I` (isort) rules.
   - **MyPy** (`[tool.mypy]`): `python_version = "3.11"`, `strict = true`.

### Packaging Notes
- Single `pyproject.toml` locks all dependency versions for the entire monorepo.
- No per-subpackage `setup.py` or `pyproject.toml` files; all subpackages share the top-level distribution.
- Optional dependency groups (`dev`, `cuda`, `mps`) are declared under `[project.optional-dependencies]`.

---

*Analysis generated from README.md, pyproject.toml, and `src/accentedge/` directory tree.*
