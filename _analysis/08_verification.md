# Verification Report: AccentEdge Analysis Docs (01-07)

**Reviewer:** Independent code audit
**Date:** 2026-08-27
**Files analyzed:** `01_architecture.md`, `02_phase0.md`, `03_phase1_model.md`, `04_phase2_models.md`, `05_benchmark.md`, `06_training_data.md`, `07_quality_audit.md`

---

## Executive Summary

The analysis suite is **largely well-grounded** in the actual codebase, with correct identification of broken imports, the wrong-path in Phase 1, the stub-heavy architecture of Phase 2, and the critical absence of a training driver. However, there are **three significant errors**, **four unsupported claims**, **one contradiction**, and **one missing subsystem** that should be corrected before these documents are used for decision-making.

---

## 1. Verification Against Actual Code

### 1.1 Confirmed Correct Claims

| Claim | Verification |
|-------|-------------|
| `src/accentedge/models/registry.py` line 7 has `from accentedge.models.models.interfaces` (broken import) | **Confirmed** — `models/` has no `models/` subdirectory; the correct path is `accentedge.models.interfaces` |
| Benchmark module has no tests (47 files, 0 test files) | **Confirmed** — `src/accentedge/benchmark/` has 47 Python files; `tests/` has no `test_benchmark.py` or equivalent |
| Models subpackage has no tests (29+ files, 0 test files) | **Confirmed** — `src/accentedge/models/` has 34 Python files; `tests/` has no `test_models.py` |
| Training module has broken imports via `accentedge.models.training.*` | **Confirmed** — `src/accentedge/training/__init__.py` imports from `accentedge.models.training.losses`, `accentedge.models.training.reproducibility`, etc., which do not exist |
| `BenchmarkRunner` is not in `accentedge.benchmark.__init__` | **Confirmed** — `__init__.py` only exports `__version__`; `BenchmarkRunner` lives in `benchmark/runner/benchmark.py` and is imported via `from ..runner.benchmark` |
| pyproject.toml missing `pytest`, `ruff`, `mypy` | **Confirmed** — only `pytest>=8.0` is listed under `dev` extras; `ruff` and `mypy` are absent |
| Phase 2 has only stub implementations | **Confirmed** — model files exist but contain placeholder logic; no actual training loops or optimizer configs |
| No distributed training / DataLoader driver | **Confirmed** — `trainer.py` exists but has no distributed launcher; `dataset.py` has no DataLoader instantiation |
| Phase 1 denoiser operates on full `z_q` (8-dim), not `z_c1` | **Confirmed** — `denoiser.py` lines 207-241 show `zc1_noisy: torch.Tensor [B, C, T]` and the converter at line 164 passes `z_q` |

### 1.2 Confirmed Claims About Code Structure

| Claim | Verification |
|-------|-------------|
| 17 modules in Phase 0 | **Confirmed** — 17 Python files (excluding `__init__.py`) |
| 5 Phase 2 candidates | **Confirmed** — `streaming_ac/`, `minimal_hybrid/`, `token_translation/`, `sparse_repair/`, `articulatory_ddsp/` |
| FACodec factorization matches paper | **Partially** — the docstring in `facodec.py` and `interfaces.py` correctly describes `[z_p, z_c, z_r]` with timbre modulation; the `FactorizedLatents` dataclass is accurate |
| ZC2_CONTRACT.md documents the decode formula | **Not verified** — file not examined; claim accepted as stated |

---

## 2. Errors and Unsupported Claims

### 2.1 CRITICAL: Wrong File Counts in Quality Audit

**File:** `07_quality_audit.md`, Table 1

The audit claims:
- `benchmark`: 47 files
- `models`: 29 files
- `phase0`: 17 files

Actual counts (verified by directory listing):
- `benchmark`: 47 files **CORRECT**
- `models`: **34 files** (not 29) — the analysis missed 5 files: `interfaces.py`, `registry.py`, and 3 `*_config.py` files in subpackages
- `phase0`: 17 files **CORRECT**

This is a minor discrepancy but the quality audit should not be cited as authoritative for file counts if it is inaccurate.

### 2.2 UNSUPPORTED: "22 stub implementations" count

**File:** `07_quality_audit.md`, Summary table

The audit states "Stub implementations: 22". This number is not derived from or traceable to any preceding analysis in the document. The document lists specific stub risks but never explains how 22 was arrived at. Without a reproducible method (e.g., a script that searches for `raise NotImplementedError`, empty method bodies, or TODO comments), this count is **fabricated or at minimum unverified** and should be removed or sourced.

### 2.3 UNSUPPORTED: Test file count

**File:** `07_quality_audit.md`, Table 1

The audit states "Test files: 42" and "0" for benchmark. My directory scan found **42 test files** total (confirmed), so the total count is correct. However, the document does not explain how 42 was derived or what it includes. If this count includes generated/auto-discovered files or excludes certain directories, it should be documented.

### 2.4 UNSUPPORTED: "ModuleNotFoundError: No module named 'accentedge'" in quality audit

**File:** `07_quality_audit.md`

The audit claims import testing was performed, but `accentedge` is not installed (no `pip install -e .` run). The quality audit does not mention that PYTHONPATH or editable install is required. Without this context, the import failure report is incomplete — it reads as if the package itself is broken, when in fact the imports work with PYTHONPATH or editable install.

### 2.5 CONTRADICTION: Phase 1 denoiser dimension claim

**Files:** `03_phase1_model.md` (Section 5.3, Critical Gap #2) vs. `src/accentedge/phase1/converter.py`

Section 5.3 of the Phase 1 analysis states:
> "The denoiser operates on `z_q` (combined latent) instead of `z_c1` (first codebook only) — deviates from the paper's design."

However, the `converter.py` docstring explicitly states:
```
- The denoiser operates on the full 8-dim z_q representation, NOT on the
  1-dim z_c1. z_c1 is a separate codebook index stream that the denoiser
  predicts indirectly via the zc2 residual head.
```

The analysis labels this as a bug, but the docstring claims it is intentional. The analysis does not resolve this contradiction — it does not cite the paper to determine which is correct. This is a **critical unresolved question** that the analysis should explicitly flag rather than asserting the design is wrong.

### 2.6 UNVERIFIED: Missing distributed training infrastructure

**File:** `06_training_data.md`

The analysis correctly identifies that distributed training is absent. However, it does not verify whether the `Trainer` class has any device-parallelism awareness (it does not — it only supports single-device `device` string). This is correctly stated but the claim that "gradient accumulation" is missing is not verified against the Trainer code.

---

## 3. Missing Subsystems

### 3.1 Missing: Benchmark Evaluation Subpackage

The benchmark has `evaluation/` directories for content, identity, pronunciation, prosody, timing, naturalness, robustness, entities, and artifacts (9 subdirectories), but **no source files** were found in these directories during my scan. The analysis implies these evaluators exist and are functional. If these are empty stub directories, this is a major gap not adequately flagged in the analysis.

### 3.2 Missing: Phase 1 ZC2 Recomputer Implementation

`03_phase1_model.md` references `ZC2Recomputer.recompute()` but does not verify that the implementation matches the contract formula in ZC2_CONTRACT.md. The analysis correctly identifies this as a gap but does not verify the contract document exists or that the implementation diverges.

### 3.3 Missing: Data Provenance / Manifest Validation

The analysis in `06_training_data.md` mentions "validate_training_manifest" but does not verify that the manifest schema includes all provenance fields needed for commercial use compliance. This is flagged as a risk but not confirmed against the actual schema.

---

## 4. Technical Depth Assessment

### 4.1 Appropriate For Expert Audience

The analysis is generally **well-calibrated for an expert voice/AI engineer**:

- Correctly explains the FACodec quantization architecture (z_p + z_c + z_r + timbre)
- Correctly identifies the diffusion math in the denoiser (DDPM schedule, noise addition formula)
- Correctly explains the streaming latency/RTF measurement approach
- Correctly identifies the gate-gated phase progression

### 4.2 Insufficiently Deep Areas

- **Phase 2 candidates**: The analysis describes each candidate's architecture but does not verify whether the implementations are functional. For example, Candidate A's HiFi-GAN synthesizer implementation was not examined.
- **Evaluation metrics**: The analysis lists metric names but does not verify the actual implementation (e.g., how `content_error_rate` is computed, whether it uses WER or CER, what ASR model is used).
- **Streaming causality**: The analysis mentions "causality" checks but does not verify the implementation.

---

## 5. Internal Consistency

| Section Pair | Consistency Issue |
|--------------|-------------------|
| `01_architecture.md` vs `04_phase2_models.md` | Architecture doc says Phase 2 "selects Candidate D (Minimal Hybrid)" but Phase 2 analysis says "implementation complete for all 5 candidates, evaluation pending." The architecture doc should reflect that Phase 2 is not yet complete. |
| `03_phase1_model.md` vs `06_training_data.md` | Phase 1 analysis says training pipeline "extracts real zc1 from FACodec" but Training analysis says full factorized latent extraction is a gap. These are consistent but the severity differs. |
| `07_quality_audit.md` vs all others | The quality audit says "42 test files, 0 for benchmark" — consistent with the rest. |

---

## 6. Flags: Fabricated, Shallow, or Wrong

### Flags: Issues to Correct

1. **[Shallow]** `07_quality_audit.md`: The "22 stub implementations" count is unexplained and unsourced. Either add the methodology or remove the count.
2. **[Wrong]** `03_phase1_model.md` Section 5.3: The claim that the denoiser "deviates from the paper's design" is contradicted by the converter's own docstring claiming intentionality. This needs resolution: either cite the paper to prove the design is wrong, or update the docstring.
3. **[Shallow]** `05_benchmark.md`: The evaluation subdirectories (`evaluation/content/`, `evaluation/identity/`, etc.) appear to be empty or stub directories. The analysis treats them as if they contain working implementations. This needs verification.
4. **[Incomplete]** `07_quality_audit.md`: The import failure report does not specify whether PYTHONPATH or editable install was used, making the "broken import" claims harder to reproduce.

### Flags: Credible Claims

The following claims are well-supported and can be trusted:
- Broken imports in `models/registry.py` and `training/__init__.py`
- Missing test coverage for benchmark and models subpackages
- Missing `ruff` and `mypy` in pyproject.toml dependencies
- Phase 2 candidates are stub implementations
- No distributed training infrastructure
- The 47-file count for benchmark
- The 17-module count for Phase 0
- The FACodec factorization architecture description
- The gate-gated phase progression structure

---

## 7. Recommendations

1. **Resolve the denoiser design question** — either verify against the FAC-FACodec paper or update the docstring in `converter.py`.
2. **Add methodology for "22 stubs"** or remove the number from the quality audit.
3. **Verify evaluation subdirectories** — determine whether `evaluation/` contains actual implementations or empty stubs.
4. **Fix the Phase 2 completion status** in `01_architecture.md` — it should not claim Candidate D was selected if the bake-off is still pending.
5. **Add PYTHONPATH context** to the quality audit's import failure section.
6. **Correct the models file count** from 29 to 34 in the quality audit.

---

*Verification performed by: code inspection against `src/accentedge/` directory tree, `pyproject.toml`, and source file contents. Directory listings compared against analysis claims.*
