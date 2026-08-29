# Phase 2 Streaming Candidate Models — Analysis

**Date:** 2026-08-27  
**Scope:** Architecture bake-off candidates A/B/C/D and Sparse Repair  
**Source files:** `src/accentedge/models/**/*.py`, `src/accentedge/streaming/*.py`, `docs/phase1/ARCHITECTURE_DECISIONS.md`

---

## 1. The Five Streaming Candidates

### A — Streaming AC

**Architecture**
- **Modules:** Content/prosody encoder → speaker encoder → accent bottleneck → synthesizer
- **Modes:** `paper_style` (640 ms lookahead, 4-layer encoder, 256 hidden dim) and `low_lookahead` (0 ms lookahead, 2-layer encoder, 128 hidden dim)
- **Encoder:** Linear projection + LayerNorm + N residual linear/GELU layers; operates on mel-like input frames (20 ms)
- **Speaker encoder:** 2-layer Conv1D + AdaptiveAvgPool1d + linear projection → 64-D embedding
- **Bottleneck:** Linear projection from content dim → accent latent (e.g., 256 → 64)
- **Synthesizer:** HiFi-GAN-inspired lightweight generator with 3 dilated Conv1D layers + final Conv1D; conditioned on accent latent, speaker embedding, f0, and energy

**What it does differently**
- Explicit speaker disentanglement via a dedicated speaker encoder
- Separates content, prosody, accent, and speaker into distinct modules
- Supports conversion strength and target accent conditioning
- Paper-style mode uses a large 640 ms lookahead for quality; low-lookahead mode removes it

**Strengths**
- Proven encoder–decoder separation paradigm
- Speaker embedding allows identity-aware processing
- Large hidden dimension (256) gives high representational capacity
- Four modules create a richer transformation pipeline

**Weaknesses**
- Paper-style mode’s 640 ms lookahead exceeds conversational latency budgets
- Largest parameter footprint (~4M estimated for paper style, ~1M for low-lookahead)
- Decoder buffer (`decoder_state["buffer"]`) grows unboundedly with session length unless explicitly truncated
- Encoder cache state in `StreamingACSession` can accumulate
- 4-module depth increases failure modes and training complexity

**Streaming characteristics**
- `required_lookahead_ms`: 640 (paper) / 0 (low-lookahead)
- Frame accumulation: ~80 ms (paper) / ~40 ms (low-lookahead)
- Total algorithmic latency: ~880 ms (paper) / ~120 ms (low-lookahead)

---

### B — Articulatory DDSP

**Architecture**
- **Modules:** Waveform encoder → articulatory feature mapper → DDSP harmonic+noise synthesizer
- **Encoder:** Configurable hidden dim (default 128), frame rate 10 ms
- **Mapper:** Transforms encoder features toward target accent in articulatory space
- **Synthesizer:** DDSP-based — harmonic oscillators (64) + noise bands (32) — cheap, differentiable synthesis

**What it does differently**
- Operates in an explicit articulatory/phonetic control space rather than raw acoustics
- Synthesis is synthesiser-driven (oscillators + noise), not a neural upsampler
- Very low algorithmic delay by design (0 ms lookahead)

**Strengths**
- 0 ms lookahead; strictly causal
- Physically interpretable intermediate representation
- Fast synthesis (oscillator + noise bank compute is O(1) per frame)
- If accent differences map cleanly to articulatory parameter shifts, the mapper works in a semantically meaningful space

**Weaknesses**
- DDSP synthesis has a known quality ceiling on complex accents or nuanced spectral changes
- Articulatory parameter estimation errors compound through mapper and synthesizer
- No proven large-scale training recipe for accent conversion in articulatory space
- Harmonic oscillators may not capture allophone variation, coarticulation, or prosodic accent patterns

**Streaming characteristics**
- `required_lookahead_ms`: 0
- Frame accumulation: ~10 ms
- Total algorithmic latency: ~30 ms
- State: `encoder_state`, `mapper_state`, `synth_state` dicts — bounded O(1) per step

---

### C — Token Translation

**Architecture**
- **Modules:** Causal speech tokenizer → LSTM accent translator (with FiLM) → token-conditioned synthesizer
- **Tokenizer:** Causal Conv1D → continuous soft token embeddings at 50 Hz (20 ms frames)
- **Translator:** 2-layer LSTM with FiLM conditioning by target accent; 0-frame lookahead in default config
- **Synthesizer:** Transposed conv + speaker FiLM → waveform at token rate, then upsampled

**What it does differently**
- Introduces a structured discrete-ish intermediate token representation
- Translation happens in token space, not acoustic or articulatory space
- LSTM translator with FiLM enables sequential accent conditioning

**Strengths**
- 0-frame lookahead; strict causality
- Structured intermediate representation separates concerns
- Proven token-based paradigm (inspired by PHONOS-style architectures)
- Tokenization can compress phonetic detail into a compact form

**Weaknesses**
- LSTM state grows linearly with session duration — `translator_state` accumulates hidden/cell states
- Three separate state dictionaries (`tokenizer_state`, `translator_state`, `synth_state`) increase memory burden
- Tokenizer quality is a prerequisite: lossy tokenization caps the achievable output quality
- Synthesizer upsampling from token rate to waveform may produce audible artifacts
- `count_parameters()` exists but no documented parameter target — not optimized for size

**Streaming characteristics**
- `required_lookahead_ms`: 0
- Frame accumulation: ~20 ms
- Total algorithmic latency: ~60 ms
- State growth: **LINEAR_GROWTH risk** for sessions >10 min

---

### D — Minimal Hybrid

**Architecture**
- **Modules:** Causal Conv1d encoder → per-accent affine mapper → ConvTranspose1d synthesizer
- **Encoder:** 2-layer causal Conv1D block (ConstantPad1d left-padding + Conv1d + LayerNorm + GELU), followed by `AvgPool1d` downsampling to frame rate; input: raw waveform (B, 1, T)
- **Mapper:** Single linear per-accent shift + scale embeddings (`accent_shift` and `accent_scale` as `nn.Embedding`); strength ∈ [0, 1] interpolates between identity and full accent mapping
- **Synthesizer:** Single `ConvTranspose1d` layer for learned upsampling; kernel = 2×hop, stride = hop

**What it does differently**
- Deliberately minimal: only 3 modules, linear mapper, no recurrent state
- Operates directly on raw waveform (no mel features required)
- Explicit per-accent affine transformation rather than a shared latent
- Strength control is built into the mapper formula: `mapped = features*(1−s) + (features*scale + shift)*s`

**Strengths**
- 0 ms lookahead; strictly causal (left-only padding)
- < 500K parameter target; `count_parameters()` confirms at init time
- Simplest gradient path → fewest failure modes, fastest iteration
- State growth is bounded: `_MinimalHybridSession.state_size_bytes()` = `len(timeline) * 32` bytes; O(T) with a tiny constant
- Supports `target_accent` and `conversion_strength` natively
- No speaker disentanglement needed for baseline; can be added in Phase 3

**Weaknesses**
- Linear mapper may be too simple for complex accent transformations
- ConvTranspose1d upsampler may produce audible artifacts vs. autoregressive or flow-based alternatives
- No speaker conditioning → identity preservation depends entirely on encoder retention
- Lower quality ceiling than deeper models

**Streaming characteristics**
- `required_lookahead_ms`: 0
- Frame accumulation: ~20 ms
- Total algorithmic latency: ~60 ms
- State: timeline list only; bounded growth

---

### Sparse Repair

**Architecture**
- **Modules:** Streaming deviation detector → repair controller → localized overlap-add synthesizer
- **Detector:** Simple frame-level feature extractor (mean + std) + lightweight classifier
- **Controller:** Converts `DeviationDecision` into `RepairControls` with `start_sample`, `end_sample`, `strength`, `commit_time`
- **Synthesizer:** Repairs only flagged regions using overlap-add with configurable fade

**What it does differently**
- Not a full accent-conversion model; it is a surgical post-hoc intervention layer
- Only reprocesses regions flagged as deviating from target accent
- Preserves source audio where acceptable; minimal compute per repair

**Strengths**
- Complements any candidate rather than replacing it
- Minimal compute per repair
- Preserves acceptable audio segments untouched

**Weaknesses**
- Interfaces and config scaffolded only; no trained model implemented
- Detector quality unvalidated
- Boundary artifacts at repair regions possible
- Repair artifacts unquantified

**Streaming characteristics**
- `required_lookahead_ms`: 0
- Not evaluated in Phase 2; candidate for Phase 3 enhancement

---

## 2. StreamingCandidate Protocol and StreamingSession Interface

### StreamingCandidate Protocol (`models/interfaces.py`)

```python
class StreamingCandidate(Protocol):
    metadata: CandidateMetadata

    def prepare(self, device: str, precision: str) -> None: ...
    def create_session(self, config: dict[str, Any]) -> StreamingSession: ...
    def process_chunk(
        self,
        session: StreamingSession,
        audio_chunk: np.ndarray,
        sample_rate: int,
    ) -> StreamingResult: ...
    def flush(self, session: StreamingSession) -> list[StreamingResult]: ...
    def reset(self, session: StreamingSession) -> None: ...
    def close(self) -> None: ...
```

- **`metadata`**: `CandidateMetadata` Pydantic model carrying `architecture_id`, `version`, `input_sample_rate`, `frame_ms`, `preferred_chunk_ms`, `required_lookahead_ms`, `left_context_ms`, and boolean capability flags (`supports_conversion_strength`, `supports_target_accent`, `requires_reference_speaker`, `uses_text_at_inference`), plus `parameter_count` and `commercial_use_status`.
- **`prepare`**: Moves model weights to target device/precision (cpu/cuda, fp32/fp16/bf16).
- **`create_session`**: Factory for a fresh `StreamingSession`. Config dict can carry simulator/test flags.
- **`process_chunk`**: Core streaming inference. Accepts one chunk of `np.ndarray` audio and returns a `StreamingResult`.
- **`flush`**: Drain any buffered outputs at session end.
- **`reset`**: Clear all session-internal state for reuse.
- **`close`**: Free GPU/CPU resources; mark instance closed.

### StreamingSession Interface (`models/interfaces.py`)

```python
class StreamingSession:
    def __init__(
        self,
        session_id: str,
        created_at: datetime | None = None,
        state: dict[str, Any] | None = None,
        samples_processed: int = 0,
    ) -> None: ...

    def state_size_bytes(self) -> int: ...
```

- **`session_id`**: Unique identifier for the streaming session.
- **`state`**: Opaque dictionary hosting candidate-specific state objects (e.g., `{"ac": ACState(...)}`, `{"d": _MinimalHybridSession(...)}`).
- **`samples_processed`**: Running total of input samples seen; used to compute input/output sample offsets.
- **`state_size_bytes()`**: Traverses `state` values and sums `nbytes` for `np.ndarray`, `bytes`, and nested structures. Used by `measure_state_growth`.

### StreamingResult

```python
class StreamingResult:
    def __init__(
        self,
        audio: np.ndarray,
        sample_rate: int,
        input_start_sample: int,
        input_end_sample: int,
        output_start_sample: int,
        output_end_sample: int,
        algorithmic_delay_samples: int = 0,
        metadata: dict[str, Any] | None = None,
    ) -> None: ...
```

- Carries the produced audio waveform, sample rate, and exact input/output sample indices.
- `algorithmic_delay_samples` records the candidate’s declared latency budget.
- `metadata` stores per-chunk attributes such as `conversion_strength`, `token_rate_hz`, etc.

---

## 3. ModelRegistry

**Location:** `src/accentedge/models/registry.py`

```python
class ModelRegistry:
    def __init__(self) -> None:
        self._registry: dict[str, type[StreamingCandidate]] = {}

    def register(self, arch_id: str, cls: type[StreamingCandidate]) -> None: ...
    def get(self, arch_id: str) -> type[StreamingCandidate]: ...
    def list_available(self) -> list[str]: ...
    def create(self, arch_id: str, config: dict[str, Any]) -> StreamingCandidate: ...
    def sweep_configs(self, arch_id, base_config, chunk_sizes, lookahead_values) -> list[dict]: ...
    def create_for_configs(self, arch_id, configs) -> list[StreamingCandidate]: ...
```

**Design notes**
- Singleton via `get_registry()` with module-level `_registry` cache.
- `register` enforces unique architecture IDs.
- `create` instantiates via `cls.__new__(cls)` (bypasses `__init__`), then patches `metadata.architecture_id` if the instance already has a metadata object.
- `sweep_configs` imports `CHUNK_SIZES_MS` and `LOOKAHEAD_SIZES_MS` from `accentedge.benchmark.sweeps` to generate Cartesian product configs.
- `create_for_configs` is a convenience wrapper for sweep-based evaluation.

---

## 4. Virtual-Time Streaming Simulator (`simulator.py`)

### How it works

```python
class StreamingSimulator:
    def __init__(self, candidate, chunk_size_ms=80, lookahead_ms=0, sample_rate=16000): ...
    async def feed(self, audio: np.ndarray) -> list[StreamingResult]: ...
    def report(self) -> SimulatorReport: ...
```

1. **Session creation:** Instantiates a `StreamingSession` via `candidate.create_session({"simulator": True})`.
2. **Chunking:** Delegates to `Chunker(chunk_size_ms, sample_rate, overlap_ms=lookahead_ms)`. Overlap is driven by the declared lookahead.
3. **Virtual-time loop:**
   - Tracks `virtual_time_ms` (wall-clock time of the input stream) and `backlog_ms` (processing latency).
   - For each chunk, emits `SimulatorEvent` records: `chunk_arrival`, `processing_start`, `processing_end`, `output_ready`.
   - Measures per-chunk compute time with `asyncio.get_event_loop().time()`.
   - Updates backlog: `backlog_ms = max(0, backlog_ms + elapsed_ms − chunk_duration_ms)`.
   - Advances `virtual_time_ms` by the chunk’s audio duration.
4. **Flush/close:** Calls `candidate.flush(session)` in a `finally` block, then `candidate.close()`.

### Metrics produced (`SimulatorReport`)

| Metric | Meaning |
|--------|---------|
| `latency_p50_ms`, `p95_ms`, `p99_ms` | Per-chunk processing time (end_timestamp − start_timestamp) |
| `max_backlog_ms` | Peak backlog across all chunks |
| `avg_backlog_ms` | Mean backlog across all events |
| `total_chunks` | Number of chunks processed |
| `total_output_samples` | Sum of `result.audio.size` across all returned results |
| `rtf_p50`, `rtf_p95`, `rtf_p99` | Real-time factor = `compute_ms / audio_ms` per chunk |

**Note on RTF calculation:** In the current implementation, `audio_ms` and `compute_ms` are computed from the same timestamp delta because `processing_end − processing_start` equals the wall-clock compute time, while the audio duration is `chunk_duration_ms`. The formula is effectively `elapsed_ms / chunk_duration_ms`.

---

## 5. Chunker and Timeline

### Chunker (`chunker.py`)

```python
class Chunker:
    def __init__(self, chunk_size_ms: int, sample_rate: int, overlap_ms: int = 0): ...
    @property
    def chunk_samples(self) -> int: ...
    @property
    def overlap_samples(self) -> int: ...
    def chunk(self, audio: np.ndarray) -> list[np.ndarray]: ...
```

- Converts `chunk_size_ms` and `overlap_ms` to sample counts.
- Iterates with stride `chunk_samples − overlap_samples`.
- Returns a list of 1-D `np.ndarray` chunks; the last chunk may be shorter.
- Used by the simulator and intended for real-time streaming pipelines.

### Timeline (`timeline.py`)

```python
class TimingMapping:
    def __init__(self, input_start, input_end, output_start, output_end): ...

class Timeline:
    def __init__(self): ...
    def add(self, input_start, input_end, output_start, output_end): ...
    def get_output_for_input(self, input_sample: int) -> int | None: ...
    def get_input_for_output(self, output_sample: int) -> int | None: ...
    def total_input_samples(self) -> int: ...
    def total_output_samples(self) -> int: ...
    def drift_samples(self) -> float: ...
```

- Records the mapping between input sample ranges and output sample ranges for each processed chunk.
- `get_output_for_input` performs linear interpolation within the mapped range to estimate where a given input sample lands in output time.
- `get_input_for_output` performs the inverse mapping.
- `drift_samples` = `total_output_samples − total_input_samples`; used to detect cumulative resampling drift.

---

## 6. Continuity Checks and State Growth Analysis

### Continuity Checks (`continuity.py`)

```python
@dataclass
class Discontinuity:
    sample_index: int
    amplitude_jump_db: float
    phase_jump: float

@dataclass
class ContinuityResult:
    discontinuities: list[Discontinuity] = ...
    max_amplitude_jump_db: float = ...
    max_phase_jump: float = ...

def measure_chunk_boundary_artifacts(
    output_audio: np.ndarray, chunk_size_samples: int
) -> ContinuityResult: ...
```

- Samples every `chunk_size_samples`-th boundary in the output audio.
- Computes `amplitude_jump_db` between the sample before and after each boundary.
- A jump of `inf` is recorded when crossing through or near zero (division-by-zero guard).
- `phase_jump` is currently always 0.0 (placeholder for future phase analysis).
- These metrics quantify chunk-boundary artifacts introduced by the streaming process (e.g., overlap-add misalignment, causal padding edge effects).

### State Growth Analysis (`state_growth.py`)

```python
@dataclass
class StateGrowthResult:
    grew: bool = False
    sizes: list[tuple[int, int]] = ...   # (total_samples_produced, state_bytes)
    growth_rate: float = 0.0             # bytes per second
    verdict: Literal["BOUNDED", "LINEAR_GROWTH", "UNKNOWN"] = "UNKNOWN"

async def measure_state_growth(
    candidate_factory,
    duration_seconds: int = 1800,
    sample_rate: int = 16000,
) -> StateGrowthResult: ...
```

- Simulates a 30-minute session by feeding 0.1-second zero chunks.
- Records `session.state_size_bytes()` at 1 min, 10 min, 30 min checkpoints.
- Computes `growth_rate = (final_size − initial_size) / elapsed_seconds`.
- **Threshold:** > 1000 bytes/sec → `LINEAR_GROWTH`; otherwise → `BOUNDED`.
- **Findings from code review:**
  - **A (paper_style / low_lookahead):** `StreamingACSession.state_size_bytes()` includes `encoder_cache`, `speaker_state["embedding"]`, and `decoder_state["buffer"]`. The decoder buffer accumulates `audio_np` arrays every chunk (`ac_state.decoder_state["buffer"].append(audio_np)`) and is only drained on `flush`. In a long-running session without periodic flush, this is unbounded growth.
  - **B (Articulatory DDSP):** Session state is plain dicts (`encoder_state`, `mapper_state`, `synth_state`); `state_size_bytes()` is not explicitly implemented on the session, so the generic `StreamingSession.state_size_bytes()` traverses values. Since encoder/mapper/synth states are scalars or small tensors, growth is bounded.
  - **C (Token Translation):** `TokenTranslationSession.state_size_bytes()` sums numpy arrays in three dicts. However, LSTM hidden/cell states grow with sequence length if stored naïvely, and the `timeline` list also grows. High risk of `LINEAR_GROWTH`.
  - **D (Minimal Hybrid):** `_MinimalHybridSession.state_size_bytes()` = `len(self.timeline) * 32`. Timeline entries grow linearly but with a very small constant (32 bytes per chunk). For 30 min at 80 ms chunks: ~22,500 chunks × 32 B ≈ 720 KB. Well below 1000 bytes/sec threshold (≈ 24 B/sec).
  - **Sparse Repair:** `pending_repairs` list accumulates `DeviationDecision` objects; pending repairs are eventually committed or cleared at flush, but in steady-state streaming the list can grow if repairs are deferred. Risk of bounded-but-non-trivial growth.

---

## 7. Why Candidate D Was Selected

### Decision summary

**Candidate D — Minimal Hybrid** is the chosen architecture for Phase 3 advancement, recorded in `docs/phase1/ARCHITECTURE_DECISIONS.md`.

### Rationale (from ADR + code evidence)

1. **Passes all hard gates by design**
   - `required_lookahead_ms = 0`
   - Strictly causal Conv1D (left-only `ConstantPad1d`)
   - Bounded state: `_MinimalHybridSession.state_size_bytes()` grows linearly with a tiny constant (32 B per chunk)
   - Prefix invariance: no future-dependent operations; `process_chunk` only sees the current chunk

2. **Simplest viable gradient path**
   - Only 3 modules: encoder → mapper → synthesizer
   - No LSTM recurrence, no speaker encoder, no tokenizer
   - Fewer failure modes and easier to debug in Phase 3

3. **Smallest footprint**
   - `count_parameters()` implemented; target < 500K parameters
   - 2-layer causal Conv1D (64 hidden) + linear mapper (64-D embeddings) + single ConvTranspose1d upsampler
   - Compare: A low-lookahead estimated ~1M+; B ~1.5M; C ~2M

4. **Explicit conversion strength control**
   - Mapper formula `mapped = features * (1−s) + (features*scale + shift)*s` gives smooth, bounded interpolation
   - Strength ∈ [0, 1] is a first-class citizen, not an afterthought

5. **Per-accent embeddings**
   - `accent_shift` and `accent_scale` are `nn.Embedding(num_accents, hidden_dim)`
   - Model learns an explicit per-accent transformation rather than mixing all accents into a shared latent

6. **Lowest Phase 3 risk**
   - Architecture is fully specified and minimal
   - Phase 3 can focus on quality improvements (deeper mapper, better losses, speaker conditioning) rather than debugging architectural bugs

### Scored comparison (ADR design-phase estimates)

| Criterion (weight) | A (paper) | A (low-look) | B | C | **D** |
|--------------------|-----------|--------------|---|---|-------|
| Streaming latency (20%) | 1/5 | 3/5 | 5/5 | 4/5 | **5/5** |
| Quality potential (20%) | 4/5 | 3/5 | 3/5 | 3/5 | **3/5** |
| Parameter efficiency (20%) | 2/5 | 3/5 | 3/5 | 3/5 | **5/5** |
| Implementation risk (20%) | 2/5 | 3/5 | 3/5 | 2/5 | **5/5** |
| Phase 3 extensibility (20%) | 3/5 | 3/5 | 3/5 | 4/5 | **5/5** |
| **Score** | **2.4** | **3.0** | **3.4** | **3.2** | **4.6** |

### Why the others were rejected

**Candidate A (Streaming AC)**
- Paper-style mode: 640 ms lookahead violates conversational latency budget.
- Low-lookahead mode: still carries 4 modules (~1M+ parameters). Marginal benefit over D does not justify complexity.
- Decoder buffer (`decoder_state["buffer"]`) grows unboundedly in long sessions without explicit truncation.

**Candidate C (Token Translation)**
- LSTM translator state grows linearly with session duration; `TokenTranslationSession.state_size_bytes()` reveals risk of `LINEAR_GROWTH` verdict.
- Tokenizer quality is a prerequisite; if the tokenizer loses phonetic detail, the translator cannot recover it.
- `count_parameters()` exists but no parameter target is documented, suggesting it was not optimized for size.

**Candidate B (Articulatory DDSP)**
- Passes streaming gates (0 ms lookahead, causal encoder, bounded state).
- Retained as **backup** architecture.
- Rejected for primary advancement due to DDSP quality-ceiling risk and lack of proven large-scale training recipe for accent conversion.

**Sparse Repair**
- Not evaluated: only interfaces (`StreamingDeviationDetector`, `RepairController`, `SparseSynthesizer`) and config exist; no trained model.
- Better positioned as a Phase 3 enhancement layer on top of D rather than a standalone candidate.

### Known risks for D in Phase 3

| Risk | Mitigation |
|------|------------|
| Linear mapper too simple for quality targets | Deepen mapper to 2–3 layer MLP or light transformer; keep causal |
| ConvTranspose1d upsampler produces audible artifacts | Replace with windowed overlap-add or small LSTM post-filter |
| No speaker disentanglement → identity loss | Add speaker encoder + FiLM conditioning in mapper |
| Benchmark numbers pending full training | ADR is design-phase; post-training results will populate placeholders |

---

## Appendix: Key File Map

| Concept | Primary file(s) |
|---------|-----------------|
| Protocol + metadata | `src/accentedge/models/interfaces.py` |
| Candidate A | `src/accentedge/models/streaming_ac/streaming_ac.py`, `paper_style.py`, `low_lookahead.py`, `state.py` |
| Candidate B | `src/accentedge/models/articulatory_ddsp/articulatory_candidate.py`, `encoder.py`, `mapper.py`, `ddsp_synth.py` |
| Candidate C | `src/accentedge/models/token_translation/token_translation_candidate.py`, `tokenizer.py`, `translator.py`, `synthesizer.py` |
| Candidate D | `src/accentedge/models/minimal_hybrid/model.py` |
| Sparse Repair | `src/accentedge/models/sparse_repair/sparse_repair_candidate.py`, `detector.py`, `controller.py`, `synthesizer.py` |
| Registry | `src/accentedge/models/registry.py` |
| Simulator | `src/accentedge/streaming/simulator.py` |
| Chunker | `src/accentedge/streaming/chunker.py` |
| Timeline | `src/accentedge/streaming/timeline.py` |
| Continuity | `src/accentedge/streaming/continuity.py` |
| State growth | `src/accentedge/streaming/state_growth.py` |
| Causality | `src/accentedge/streaming/causality.py` |
| ADR | `docs/phase1/ARCHITECTURE_DECISIONS.md` |
