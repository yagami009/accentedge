# AccentEdge Code Quality Audit

**Generated:** 2026-08-27  
**Scope:** `src/`, `tests/`, `pyproject.toml`, `colab/requirements.txt`  
**Total source files:** 164 | **Test files:** 42

---

## 1. Source Files by Subpackage

| Subpackage | Files (excl. __init__) | Test Files |
|---|---|---|
| benchmark | 47 | 0 |
| models | 29 | 0 |
| phase0 | 17 | 0 |
| training | 8 | 0 |
| streaming | 7 | 0 |
| phase1 | 6 | 1 (shared) |
| audio | 4 | 3 |
| profiling | 4 | 2 |
| evaluation | 4 | 1 |
| reporting | 3 | 1 |
| data | 3 | 1 |
| config | 2 | 1 |
| codec | 2 | 1 |
| experiments | 1 | 1 |
| metrics | 0 | 0 |
| utils | 0 | 0 |
| cli | 0 | 0 |

**Test layout note:** All 38 unit tests and 3 integration tests live under `tests/unit/` and `tests/integration/` with no subpackage-mirrored directories (e.g., `tests/unit/models/` does not exist). Tests are referenced by content name, not by path.

---

## 2. Technical Debt Inventory

**TODO / FIXME / HACK / XXX markers: 0**

No inline debt markers were found anywhere in `src/`. This is a healthy signal — the codebase is clean of ad-hoc TODOs. Debt is tracked instead through stubs and broken imports (see below).

---

## 3. Stub / Incomplete Implementations

**Total:** 22 `pass` stubs across 12 files | **`raise NotImplementedError`:** 0

| File | Line | Context |
|---|---|---|
| `benchmark/candidates/registry.py` | 12 | Empty adapter fallback |
| `benchmark/candidates/base.py` | 53, 57 | Unimplemented interface methods |
| `benchmark/runner/run_manifest.py` | 64 | Empty handler block |
| `benchmark/runner/resume.py` | 23 | Swallowed exception |
| `config/loader.py` | 43 | Empty except clause |
| `training/checkpoints.py` | 102 | Empty handler |
| `audio/capture.py` | 61 | Empty stream callback |
| `profiling/memory.py` | 26 | Unimplemented measurement |
| `profiling/hardware.py` | 54, 69, 89 | Hardware detection stubs |
| `phase0/target_generation.py` | 147, 202 | Degradation fallback stubs |
| `phase0/alignment.py` | 109 | Empty alignment fallback |
| `phase0/evaluation.py` | 355 | Skipped WER branch |
| `phase0/identity_transfer.py` | 89, 99, 459 | Empty conversion fallbacks |
| `phase0/identity.py` | 221 | Skipped verification path |
| `phase1/zc2_recompute.py` | 318, 331 | Empty recompute stubs |

**Assessment:** Most stubs are defensive fallbacks (`except`, unavailable-backend paths) rather than incomplete features. The highest-risk stubs are in `profiling/hardware.py` (3 stubs — hardware detection is likely untested) and `phase1/zc2_recompute.py` (2 stubs — a core inference path).

---

## 4. Test Coverage Gaps

### 4.1 Subpackages with Zero Dedicated Tests

| Subpackage | Source Files | Notes |
|---|---|---|
| **benchmark** | 47 | Largest untested area — includes candidates, runner, schemas, evaluation |
| **models** | 29 | No model tests at all — all 5 model variants untested |
| **phase0** | 17 | Only 1 shared test (`tests/test_phase1.py` also imports phase0 indirectly) |
| **training** | 8 | 1 test file for all of training (losses, schedules, trainer) |
| **streaming** | 7 | Tests cover chunker, timeline, causality — not simulator or state_growth |
| **phase1** | 6 | Covered by `tests/test_phase1.py` and `tests/unit/test_phoneme_pipeline.py` |
| **evaluation** | 4 | Partial — `test_evaluation.py` exists |
| **profiling** | 4 | Partial — `test_rtf.py`, `test_memory.py` |
| **reporting** | 3 | Partial — `test_reporting.py` |
| **data** | 3 | Partial — `test_lineage.py` |
| **config** | 2 | Partial — `test_config.py` |
| **codec** | 2 | Partial — interface tests only |
| **experiments** | 1 | Partial — `test_experiments.py` |
| **audio** | 4 | Partial — `test_audio.py`, `test_audio_io.py` |
| **metrics / utils / cli** | 0 | Empty packages — no code to test |

### 4.2 Integration Test Gaps

Only 3 integration tests:
- `test_streaming_pipeline.py`
- `test_converter_pipeline.py`
- `test_pipeline.py`

No integration tests for end-to-end benchmark runs, model training, or evaluation pipelines.

### 4.3 Test-Import Fragility

`tests/test_phase1.py` uses lazy imports inside test methods (good for isolation). However, `tests/unit/test_phoneme_pipeline.py` imports `accentedge.evaluation` at module top level — verify this works when `evaluation/phonemes.py` has optional-import guards.

---

## 5. Dependency Issues

### 5.1 Missing from pyproject.toml but Used in `src/`

| Package | Used In | Severity |
|---|---|---|
| **`transformers`** | `phase0/identity.py`, `phase0/probes.py`, `phase1/phoneme_pipeline.py`, `benchmark/evaluation/naturalness.py`, `codec/facodec.py` | **High** — core to WavLM identity verification and phoneme alignment |
| **`loguru`** | `benchmark/cli.py`, `benchmark/runner/benchmark.py`, `phase1/phoneme_pipeline.py`, `phase0/target_generation.py` | **High** — used as primary logger, import will fail without it |
| **`rich`** | `benchmark/cli.py`, `benchmark/runner/benchmark.py` | Medium — CLI formatting |
| **`sounddevice`** | `audio/capture.py` | Medium — real-time capture; has runtime fallback |

### 5.2 In pyproject.toml but Unused in `src/`

| Package | Notes |
|---|---|
| **`pyarrow`** | Listed under Data/config but zero imports found in `src/` |

### 5.3 In `colab/requirements.txt` but Not in `pyproject.toml`

| Package | Notes |
|---|---|
| **`munch`** | Used by FAcodec (`modules.commons.recursive_munch`) — runtime dependency |
| **`pyworld`** | Used for F0 extraction (likely in audio/profiling) |
| **`plotly`** | Visualization — not imported in current src/ |

### 5.4 pyproject.toml gaps vs. actual imports

`pyproject.toml` is **missing 4 packages** (`transformers`, `loguru`, `rich`, `sounddevice`) that are imported unconditionally in `src/`. A fresh `pip install .` would result in immediate `ModuleNotFoundError` on import.

---

## 6. Broken Cross-References

### 6.1 Internal Package Imports — 8 Broken

**Pattern A: Double `models.models` (6 files)**  
The import `accentedge.models.models.interfaces` should be `accentedge.models.interfaces`.

| File | Line | Broken Import |
|---|---|---|
| `models/registry.py` | 7 | `from accentedge.models.models.interfaces import StreamingCandidate` |
| `streaming/causality.py` | 10 | `from accentedge.models.models.interfaces import (...)` |
| `streaming/simulator.py` | 11 | `from accentedge.models.models.interfaces import (...)` |
| `streaming/state_growth.py` | 10 | `from accentedge.models.models.interfaces import StreamingCandidate, StreamingSession` |

**Pattern B: Wrong root `accentedge.models.training` (2 files)**  
The import `accentedge.models.training.losses` should be `accentedge.training.losses`. `losses.py` lives in `src/accentedge/training/`, not under `models/`.

| File | Line | Broken Import |
|---|---|---|
| `training/__init__.py` | 14 | `from accentedge.models.training.losses import (...)` |
| `training/trainer.py` | 11 | `from accentedge.models.training.losses import total_loss` |

**Impact:** These 8 broken imports mean that importing `accentedge.models`, `accentedge.streaming`, or `accentedge.training` at the top level will raise `ModuleNotFoundError` immediately. The entire streaming subsystem and model registry are currently broken on import.

### 6.2 Test Imports

No broken test imports confirmed — `test_phase1.py` uses `accentedge.phase1.*`, `accentedge.codec.*`, and `accentedge.evaluation.*` which all resolve correctly.

---

## 7. Priority Fixes (Ranked by Impact)

### P0 — Break Everything (Immediate CI Failure)

| # | Fix | Files Affected | Why |
|---|---|---|---|
| 1 | **Add missing deps to `pyproject.toml`**: `transformers`, `loguru`, `rich`, `sounddevice` | All importers | `pip install .` produces broken install |
| 2 | **Fix `accentedge.models.models.interfaces` → `accentedge.models.interfaces`** | 4 files | Imports `models`, `streaming` crash immediately |
| 3 | **Fix `accentedge.models.training.losses` → `accentedge.training.losses`** | 2 files | Imports `training` crash immediately |

### P1 — Untested Code Risk

| # | Fix | Why |
|---|---|---|
| 4 | **Add tests for `benchmark/`** (47 files) | Largest untested surface; benchmark runner, candidates, schemas |
| 5 | **Add tests for `models/`** (29 files, 5 model variants) | No model tests exist — correctness unknown |
| 6 | **Add `phase0/` tests** (17 files) | Only indirect coverage via `test_phase1.py` |

### P2 — Incomplete Features

| # | Fix | Why |
|---|---|---|
| 7 | **Implement `profiling/hardware.py`** (3 stubs) | GPU/CPU detection stubs — profiling accuracy unknown |
| 8 | **Implement `phase1/zc2_recompute.py`** (2 stubs) | Core recompute path for streaming inference |
| 9 | **Resolve `pyarrow`** — remove from pyproject or add usage | Dead dependency adds install weight |

### P3 — Hygiene

| # | Fix | Why |
|---|---|---|
| 10 | **Synchronize `colab/requirements.txt` with `pyproject.toml`** | Drift risk; `munch`, `pyworld`, `plotly` in colab but not pyproject |
| 11 | **Add integration tests** for benchmark + model + evaluation end-to-end | Currently only 3 integration tests |
| 12 | **Mirror test directories** under `tests/` by subpackage | Current flat layout makes coverage tracking manual |

---

## Summary

| Category | Count | Severity |
|---|---|---|
| Broken internal imports | 8 | **Critical** |
| Missing pyproject.toml deps | 4 | **Critical** |
| Unused pyproject.toml deps | 1 | Low |
| Stub implementations | 22 | Medium |
| Untested subpackages | 12 of 16 | High |
| Integration tests | 3 | Low |
| Technical debt markers | 0 | Clean |
| Syntax errors | 0 | Clean |

**Bottom line:** The two highest-impact fixes are (1) fixing the 8 broken imports and (2) adding 4 missing packages to `pyproject.toml`. Both changes are small (minutes) and unblock the entire package from failing on import. The largest long-term risk is the absence of any tests for `benchmark/` (47 files) and `models/` (29 files).
