# AccentEdge Forensic Engineering Audit

**Date:** 2026-08-25
**Branch:** accent-edge-p0
**Commits audited:** a52e34a -> c8adcbd (11 commits)

---

# Executive Verdict

AccentEdge contains a genuine working offline accent conversion prototype built through 40+ iterative debugging runs. However, it is NOT a real-time system, NOT a streaming system, and has NOT validated the core product thesis.

**What was built well:** Reverse-engineering Seed-VC's undocumented API. Persistence through failures to reach working conversion.

**Biggest misconception:** "P0 complete" was never accurate. P0 infrastructure exists but P0 runtime validation has not been achieved.

**Current state:** 5 successful full-utterance offline conversions on CPU producing WAV files (runs 028, 030, 038, 039, 040). Never demonstrated streaming. Never used MPS despite availability. Never validated accent shift vs. speaker copying.

**Primary technical risk:** Seed-VC is a zero-shot voice converter, NOT an accent converter. Using a US-English reference risks transforming the source into a US speaker rather than an Indian speaker with US pronunciation. This unproven assumption could invalidate the entire product thesis.

---

# 1. Repository Map

## Source Files

```
src/
├── __init__.py
├── api/
│   ├── app.py                    # FastAPI REST + WebSocket (872 lines)
│   ├── websocket_handler.py      # WebSocketStreamHandler (NEW)
│   └── websocket_models.py       # Pydantic models (NEW)
├── audio/
│   ├── capture.py                # sounddevice capture, 16000 Hz default
│   ├── playback.py               # sounddevice playback + crossfade
│   ├── buffer.py                 # RingBuffer (thread-safe circular)
│   ├── vad.py                    # Silero VAD wrapper (untested)
├── conversion/
│   ├── seedvc.py                 # SeedVCConverter (569 lines)
│   └── engine.py                 # ConversionEngine wrapper
├── evaluation/
│   └── evaluator.py              # STOI, PESQ, MCD metrics
├── metrics/
│   ├── latency.py                # LatencyStats
│   ├── realtime_factor.py        # RTF calculation
│   └── system.py                 # psutil system monitor
├── pipeline.py                   # Main offline pipeline
├── streaming/
│   ├── __init__.py
│   ├── pipeline.py               # StreamingPipeline (scaffolded)
│   ├── vad.py                    # VADProcessor (NEW)
│   ├── websocket_audio.py        # AudioProtocol, ChunkAssembler (NEW)
│   └── latency_measurer.py       # LatencyMeasurer
```

## Test Files

```
tests/
├── test_api.py          # FastAPI endpoint tests
├── test_audio.py        # Audio module tests
├── test_conversion.py   # SeedVCConverter tests
├── test_evaluation.py   # Evaluator tests
├── test_metrics.py      # Metrics tests
├── test_streaming.py    # StreamingPipeline tests
├── test_vad.py          # VAD tests (37 tests, all pass without real model)
├── test_websocket.py    # WebSocket tests (52 tests, timeout issues)
```

## Scripts

```
scripts/
├── run_offline_conversion.py  # CLI conversion entry point
├── run_parameter_sweep.py     # Parameter optimization
├── compute_composite_score.py # Scoring script
├── run_evaluation.py          # Evaluation runner
├── run_streaming_latency.py   # Latency measurement
├── test_real_model_load.py    # Model load verification
├── run_api.py                 # API server launcher
├── record_test.py             # Recording test
├── gate_1a_whisper.py         # Whisper validation (experimental)
├── gate_1b_xtts.py            # XTTS experiment (dead)
└── gate_1c_hifigan.py         # HiFi-GAN experiment (dead)
```

## Benchmarks

```
benchmarks/
├── baseline_test.py        # Benchmark scaffold (never run)
├── latency_benchmark.py    # Latency benchmark scaffold (never run)
└── sentences.py            # Test sentence corpus
```

## Results (Runtime Artifacts)

```
results/
├── model_load_report.json      # Model load test (SUCCESS)
├── offline/
│   └── run_001..040/           # 40 conversion attempts
│       ├── run_028..040/       # 5 successful runs
│       │   ├── source.wav      # 12s Indian English
│       │   ├── reference.wav   # 12s US English
│       │   ├── output.wav      # Converted output
│       │   ├── metrics.json    # Timing + RTF data
│       │   └── run.log         # Execution log
│       └── run_001..027/       # 27 FAILED runs (various errors)
├── evaluation/                  # 5 eval JSON files
└── streaming/                   # Empty
```

## Submodule

```
seed-vc/                          # guangzhouda/seed-vc cloned locally
├── convert_voice.py              # Upstream entry point
├─��� inference.py                  # Upstream inference script
├── inference_v2.py               # Upstream v2 inference
├── seed_vc_wrapper.py            # Upstream wrapper class
├── configs/presets/              # YAML model configs
├── modules/                      # Model architectures
├── output/                       # Upstream output directory
└── converted_results/            # Upstream results directory
```

## Dependencies (requirements.txt)

Key packages (ALL unpinned unless noted):
- torch 2.2.2 (installed, not in requirements.txt)
- torchaudio (installed, not in requirements.txt)
- numpy
- soundfile / librosa (for audio I/O)
- sounddevice (for capture/playback)
- fastapi / uvicorn (for REST API)
- pyyaml
- psutil
- bigvgan (unpinned)
- campplus / funasr (unpinned)
- silero-vad (not in requirements.txt)
- no version pins on ANY dependency

## Dependencies (seed-vc/requirements.txt)

- torch
- torchaudio
- librosa
- soundfile
- gradio
- transformers
- modelscope
- vector_quantize_pytorch
- csplit
- no version pins

---

# 2. Entry Points and Call Graph

## User-Facing Entry Points

### Primary: Offline Conversion

```
scripts/run_offline_conversion.py
  -> src/conversion/engine.py  (ConversionEngine)
    -> src/conversion/seedvc.py  (SeedVCConverter)
      -> seed-vc/ modules  (model components)
      -> temp WAV files
      -> output WAV file
```

### Secondary: REST API

```
scripts/run_api.py
  -> src/api/app.py  (FastAPI app)
    -> /convert endpoint -> ConversionEngine -> SeedVCConverter
    -> /stream endpoint -> StreamingPipeline (scaffolded)
    -> /evaluate endpoint -> Evaluator
    -> /health endpoint
    -> /ws/stream -> WebSocketStreamHandler (NEW)
    -> /ws/stream/{session_id} -> WebSocketStreamHandler with reference (NEW)
```

### Tertiary: Evaluation

```
scripts/run_evaluation.py
  -> src/evaluation/evaluator.py
    -> STOI / PESQ / MCD metrics
```

## Verified Call Graph (Offline Conversion)

```
run_offline_conversion.py:main()
  -> SeedVCConverter.__init__()       # Lazy init, no model load
  -> converter.convert(source, reference, ...)
    -> _load_audio() -> soundfile.read # WAV -> numpy
    -> _ensure_model_loaded()         # Lazy model load
      -> _load_model()
        -> sys.path.insert(seed-vc/)  # Dynamic path injection
        -> yaml.safe_load(config)     # Config parsing
        -> modules.commons.build_model()
        -> load_checkpoint()
        -> SemanticEncoder init
        -> CAMPPlus init
        -> BigVGAN init
        -> RMVPE init
        -> mel_fn init
    -> _run_inference(source, reference)
      -> torchaudio.functional.resample (to 16kHz)
      -> semantic_fn(source_16k)      # Content features
      -> semantic_fn(target_16k)      # Reference content
      -> torchaudio.compliance.kaldi.fbank # Mel features
      -> CAMPPlus(target_16k)         # Style embedding
      -> RMVPE(F0)                     # Pitch extraction
      -> model.cfm.inference(cond, ...) # CFM diffusion
      -> bigvgan(mel)                  # Vocoder
      -> vc_wave.squeeze().cpu().numpy() # Output
    -> _load_audio(output_path)        # Read back WAV
    -> return {"output_audio": array, "rtf": ...}
```

**VERIFIED:** This is the actual call graph. Executed successfully 5 times (runs 028, 030, 038, 039, 040).

---

# 3. Module-by-Module Audit

## 3.1 src/conversion/seedvc.py — CRITICAL MODULE

### Purpose
Wraps the Seed-VC repository for offline voice conversion.

### Actual Implementation
- 569 lines
- Lazy model loading via `_ensure_model_loaded()` / `_load_model()`
- sys.path injection: `sys.path.insert(0, str(seedvc_path))`
- Dynamic imports: `from modules.commons import build_model`, `from modules.extractors import F0_Extractor`, etc.
- YAML config parsing
- 5 model components initialized: main model, semantic encoder, CAMPPlus style extractor, BigVGAN vocoder, RMVPE F0 extractor
- Single-pass CFM inference (no chunked generation)
- Temporary WAV files for audio I/O
- Device: auto-detect CUDA > MPS > CPU

### What Works
- Model loading: VERIFIED (9s on CPU, 1241 MB RAM)
- Checkpoint download from HuggingFace: VERIFIED
- Config resolution: VERIFIED
- Full-utterance inference: VERIFIED (5 successful runs)
- Output WAV generation: VERIFIED
- Model unloading: VERIFIED
- RTF calculation: VERIFIED (RTF = 0.003-0.004 on CPU)

### What Does NOT Work
- **MPS device:** Never used despite being available. All runs use CPU.
- **Streaming inference:** Not implemented. convert() processes entire utterance.
- **Chunked inference:** Not implemented. `_run_inference()` is utterance-based.
- **stream_output:** Not used. AccentEdge calls `cfm.inference()` directly, not `convert_voice_with_streaming()`.

### Bugs Found

**BUG-001 (P0): Gradient leak in _run_inference()**
- Location: line 524
- `vc_wave.squeeze().cpu().numpy()` fails with "Tensor requires grad"
- Fixed in run_028+ by using `.detach()` internally in upstream wrapper
- Confidence: HIGH (seen in run_027 traceback, resolved in run_028)

**BUG-002 (P1): RTF calculation includes model load time**
- Location: line 374-377
- `latency = time.time() - t0` where t0 starts BEFORE model loading
- Reported RTF of 0.003-0.004 is misleading — conversion-only RTF would be even lower
- But it means the first call's RTF includes ~9s load time
- Confidence: HIGH (directly visible in code)

**BUG-003 (P1): Temp file cleanup is racy**
- Location: lines 397-402
- Cleanup in `finally` block checks `isinstance(source_audio, np.ndarray)` to decide whether to delete
- If the same path is reused concurrently, this could delete files still in use
- Confidence: MEDIUM (code inspection)

**BUG-004 (P2): Hardcoded 16kHz resampling**
- Location: line 421-422
- `torchaudio.functional.resample(source_tensor, sr, 16000)`
- Silero VAD and CAMPPlus both expect 16kHz
- This is correct for those components but should be documented
- Confidence: HIGH

**BUG-005 (P2): _load_model can be called multiple times**
- Location: _ensure_model_loaded() uses `if self._model is not None: return`
- But _load_model() itself doesn't check before loading
- If called concurrently, could load model twice
- Confidence: MEDIUM (code inspection)

### Risks

**RISK-001 (P0): Model-architecture mismatch for accent conversion**
Seed-VC uses zero-shot voice conversion. The tiny model (DiT_uvit_tat_xlsr_ema) is designed to convert voice timbre, not accent. There is no evidence in the code, upstream docs, or runtime that it modifies pronunciation patterns while preserving Indian speaker identity. It likely copies the US reference speaker's characteristics.

**RISK-002 (P1): All runs on CPU**
MPS is available but never used. CPU inference at 37-47 seconds for 12 seconds of audio means RTF on CPU is ~0.003 (faster than real-time in terms of ratio, but with 37s of wall-clock latency). The actual user-perceived latency for a 12-second utterance is 37 seconds. For real-time, this is unacceptable.

**RISK-003 (P1): Full-utterance requirement**
The model processes entire utterances. Minimum context is unknown from the code. The CFM inference takes 37-47 seconds for 12 seconds of input on CPU. This cannot be chunked without understanding the model's minimum context window.

### Technical Debt
- sys.path injection is fragile
- No interface abstraction — ConversionEngine is a thin wrapper
- `_load_audio()` uses soundfile which has limited format support
- No caching of reference embeddings (speaker embedding recomputed every call)

---

# 4. Seed-VC Deep Audit

## 4.1 Upstream API Match

| AccentEdge Assumption | Upstream Reality | Match? | Risk |
|---|---|---|---|
| seed_vc package | No — flat imports at repo root | YES (handled) | LOW |
| SeedVCWrapper class | Exists in seed_vc_wrapper.py | YES | LOW |
| convert_voice() method | Exists in wrapper | NOT USED | MEDIUM |
| convert_voice_with_streaming() | Exists in wrapper | NOT USED | HIGH |
| YAML config | configs/presets/*.yml | YES | LOW |
| build_model() | modules.commons.build_model | YES | LOW |
| load_checkpoint() | modules.mel_processing | YES | LOW |
| semantic encoder | models/semantic | YES | LOW |
| vocoder | BigVGAN | YES | LOW |
| style encoder | CAMPPlus (funasr) | YES | LOW |
| F0 extractor | RMVPE | YES | LOW |
| stream_output parameter | True in convert_voice | NOT USED | HIGH |

## 4.2 Model Loading

Verified working. 9.19s on CPU, 1241 MB RAM peak.
All 5 components load successfully:
- model: PASS
- vocoder: PASS
- semantic_encoder: PASS
- campplus: PASS
- f0_extractor: PASS
- mel_fn: PASS

## 4.3 Inference Path

Verified working. Calls `model.cfm.inference()` with:
- cond: semantic content + mel + style + F0 conditioning
- n_steps: 10 (diffusion steps, configurable)
- No streaming capability

## 4.4 Device Selection

Order: CUDA > MPS > CPU
MPS available but NEVER selected because torch.cuda.is_available() returns False on this machine and MPS check passes but... ALL 40 runs show device=cpu. This means either MPS is not actually available or something forces CPU.

## 4.5 Precision

- All models loaded in float32
- is_half=False for RMVPE
- No autocast, no mixed precision
- On MPS, float32 is supported but slower than float16

## 4.6 Streaming Feasibility

**NOT FEASIBLE** with current architecture. The model:
1. Requires full-utterance semantic encoding (no chunked forward)
2. Uses CFM (Conditional Flow Matching) diffusion requiring n_steps per inference
3. Outputs complete waveform, not incremental samples
4. The upstream `convert_voice_with_streaming()` exists but streams post-processed output, not truly incremental inference

Minimum lookahead: unknown, likely entire utterance.

---

# 5. Audio Module Audit

## 5.1 src/audio/capture.py

### Implementation
- sounddevice.InputStream with callback interface
- Default: 16000 Hz, 200ms chunks, float32, mono
- Callback: copies indata to queue (lightweight)
- Queue: unbounded queue.Queue
- Status flags silently dropped

### What Works
- Callback does minimal work (copy + enqueue)
- Thread-safe queue delivery
- start()/stop() lifecycle

### Bugs/Risks

**BUG-006 (P1): Unbounded queue**
If consumer is slower than producer (model takes 400ms, chunks arrive every 200ms), the queue grows without bound. For a 12-second conversation, this could consume significant RAM.

**BUG-007 (P1): Status flags silently dropped**
`if status: pass` — overflow/underflow warnings are discarded. In production, this means audio corruption goes undetected.

**BUG-008 (P2): No device selection**
Always uses default input device. No way to select specific microphone.

**BUG-009 (P2): No timeout on queue read**
read_chunk(timeout=None) blocks forever if no data arrives.

## 5.2 src/audio/playback.py

### Implementation
- sounddevice.OutputStream in separate thread
- Linear crossfade between chunks (10ms default)
- Unbounded input queue
- blocksize=1024 for output stream

### Crossfade Analysis
```python
fade_out = np.linspace(1.0, 0.0, fade_len)  # linear fade out
fade_in = np.linspace(0.0, 1.0, fade_len)   # linear fade in
chunk[:fade_len] = prev_tail * fade_out + chunk * fade_in
```
This is mathematically correct for linear crossfade. Amplitude is preserved (sum = 1.0 at midpoint). No phase issues for mono. Changes output length by crossfade_samples (10ms = 220 samples at 22050 Hz).

### Bugs/Risks

**BUG-010 (P1): Unbounded queue (same as capture)**
If model produces output faster than playback consumes, queue grows unbounded.

**BUG-011 (P2): Previous tail stored by reference risk**
`self._previous_tail = chunk[-crossfade_samples:].copy()` — the .copy() is present, so this is safe.

## 5.3 src/audio/buffer.py

### Implementation
- Thread-safe ring buffer using numpy array
- Fixed capacity based on max_duration_seconds * sample_rate
- Write: copies data, wraps around if needed
- Read: returns fixed-size chunks or None
- Default: 5 seconds at 16000 Hz = 80000 samples = 320 KB

### What Works
- Thread-safe via Lock
- Correct wraparound logic
- Correct overflow behavior (returns 0 written)
- Correct underflow behavior (returns None)
- No copies on read (returns view... actually returns new array via np.zeros)
- clear() resets all pointers

### Bugs/Risks

**BUG-012 (P2): np.zeros allocation on every read_chunk()**
read_chunk() allocates a new numpy array every call. For real-time at 200ms chunks, this creates garbage collection pressure.

**BUG-013 (P2): Lock held during copy operations**
write() holds lock during entire copy. For large writes, this blocks readers.

## 5.4 src/audio/vad.py (src/audio/)

### Implementation
- VoiceActivityDetector class
- Uses Silero VAD model
- Returns VADSegment objects with start_ms, end_ms, audio
- NOT integrated into any pipeline

### Status: IMPLEMENTED BUT NOT RUNTIME-VERIFIED
No test exercises actual Silero model loading. The module exists but has never been called in a real pipeline.

---

# 6. Streaming Pipeline Audit

## 6.1 src/streaming/pipeline.py

### Implementation
- StreamingPipeline class with use_vad flag
- References AudioCapture, RingBuffer, VoiceActivityDetector, ConversionEngine, AudioPlayback
- run() method orchestrates capture → VAD → convert → playback loop

### Critical Finding: NOT WIRED

Despite importing all components, the streaming pipeline has NOT been tested end-to-end. The test file (test_streaming.py) mocks all components. No runtime verification exists.

### Connectivity Matrix

| Component | Exists | Imported | Instantiated | Used in live path |
|---|---|---|---|---|
| AudioCapture | YES | YES | YES | NOT TESTED |
| RingBuffer | YES | YES | YES | NOT TESTED |
| VAD (silero) | YES | YES | YES | NOT TESTED |
| ConversionEngine | YES | YES | YES | NOT TESTED |
| AudioPlayback | YES | YES | YES | NOT TESTED |
| LatencyStats | YES | YES | YES | SCAFFOLD |
| SystemMonitor | YES | YES | NO | NO |

## 6.2 src/streaming/vad.py

### Implementation
- VoiceActivityDetector with silero-vad + energy fallback
- VADProcessor for buffered speech segments
- Thread-safe lazy initialization

### Status: IMPLEMENTED BUT NOT RUNTIME-VERIFIED
- 37 unit tests pass (mocked)
- Real Silero model never loaded in tests
- Never integrated into live pipeline

## 6.3 WebSocket Streaming (NEW)

### src/streaming/websocket_audio.py
- AudioChunk, ChunkAssembler, AudioProtocol constants
- serialize_status/error/latency/session_end
- ChunkAssembler: fixed _max_seq_seen (security fix applied)
- 375 lines

### src/api/websocket_handler.py
- WebSocketStreamHandler with per-connection state
- Chunk validation, buffer overflow protection
- Error recovery (sends silence on failure)
- Latency tracking
- 264 lines

### src/api/app.py
- WS /ws/stream — bidirectional streaming
- WS /ws/stream/{session_id} — named session with pre-loaded reference

### Test Status
- 52 tests written
- Agent reports 50/50 passing (not independently verified by this audit)
- Integration test test_ws_stream_session_with_rest_flow requires mocking _load_audio_from_upload

---

# 7. Test Audit

## Test Classification

| Test File | Tests | Type | Mocked | Real Runtime |
|---|---|---|---|---|
| test_api.py | 15 | unit/integration | partial | FastAPI TestClient only |
| test_audio.py | 12 | unit | sounddevice mocked | NO |
| test_conversion.py | 12 | unit | model mocked | NO |
| test_evaluation.py | 8 | unit | audio files | YES (static WAVs) |
| test_metrics.py | 15 | unit | no | YES (computed values) |
| test_streaming.py | 10 | unit | ALL mocked | NO |
| test_vad.py | 37 | unit | model mocked | NO |
| test_websocket.py | 52 | unit/integration | engine/pipeline mocked | PARTIAL |

## What Tests Prove vs Don't Prove

**PROVEN by tests:**
- API endpoints return correct HTTP status codes
- Audio buffer read/write is thread-safe
- ConversionResult structure is correct
- Metrics calculations are mathematically correct
- WebSocket message serialization is correct
- Mock model loading works

**NOT PROVEN by tests:**
- Real model loads (all conversion tests mock the model)
- Real inference produces valid audio
- Real accent conversion works
- Speaker identity is preserved
- Capture + convert + playback pipeline works end-to-end
- VAD detects speech in real audio
- Streaming works with real audio
- WebSocket handles real binary audio streaming

## Skipped Tests

2 tests skipped in non-websocket suite. Need to verify which ones.

---

# 8. Performance Audit

## Verified Performance Data

From 5 successful runs (CPU only):

| Metric | Value |
|---|---|
| Model load time | 9.19s (CPU) |
| Peak RAM | 1424 MB |
| Conversion time (12s audio) | 37-47s (CPU) |
| RTF (includes load) | 0.003-0.004 |
| RAM after conversion | 186 MB (model unloaded) |
| Device | CPU (MPS never used) |

## RTF Analysis

The reported RTF of 0.003-0.004 is misleading:
- It INCLUDES model load time (9.19s)
- True inference-only RTF: conversion_time / input_duration = 37s / 12s = 3.08
- This means inference is ~3x SLOWER than real-time on CPU
- On MPS or CUDA, this would improve dramatically

## MPS Unexplained

MPS is available (torch.backends.mps.is_available() = True) but ALL runs use CPU.
Possible reasons:
1. Model components have ops not supported on MPS
2. The device resolution logic works but something forces CPU elsewhere
3. The script/test explicitly sets device="cpu"

This needs investigation — MPS could provide 5-10x speedup.

---

# 9. Sample-Rate Flow

| Component | Expected SR | Actual SR | Resamples? |
|---|---|---|---|
| Microphone (capture.py) | 16000 (default) | 16000 | N/A |
| VAD (silero) | 16000 | 16000 | N/A |
| Seed-VC model | 22050 (config) | 22050 | YES (16k->22.05k) |
| Playback (playback.py) | 16000 (default) | 16000 | YES (22.05k->16k) |
| Semantic encoder | 16000 | 16000 | torchaudio resample |
| F0 extractor | 16000 | 16000 | implicit |

**CRITICAL FINDING:** The model operates at 22050 Hz, but capture and playback default to 16000 Hz. The conversion script resamples to 22050 for model input and back to 16000 for output. This adds two resampling steps per conversion, each introducing latency and quality loss.

---

# 10. Configuration Consistency

| Parameter | Source | Centralized? |
|---|---|---|
| device | Constructor / _resolve_device() | YES |
| sample_rate | MODEL_PRESETS / constructor | PARTIAL (capture defaults 16000, model uses 22050) |
| model config | MODEL_PRESETS dict | YES |
| chunk_size | capture.py constructor | NO (per-instance) |
| crossfade_ms | playback.py constructor | NO (per-instance) |
| VAD threshold | vad.py constructor | NO (per-instance) |
| diffusion_steps | convert() parameter | NO (per-call) |

Configuration is constructor-driven, not centralized. This means different parts of the system can have mismatched sample rates without detection.

---

# 11. Security Audit

**PASS:** Path traversal in model_name was patched (commit 493eb16).
**PASS:** YAML loading uses yaml.safe_load().
**WARN:** sys.path.insert(0, str(seedvc_path)) dynamically injects the Seed-VC repo into Python path. This is necessary for flat imports but means any .py file in seed-vc/ can be imported.
**WARN:** torch.load() is used for checkpoint loading. PyTorch pickle format can execute arbitrary code. Checkpoints come from HuggingFace (Plachta/Seed-VC, funasr/campplus, lj1995/VoiceConversionWebUI).
**INFO:** tempfile is used for WAV files. Cleanup in finally block.

---

# 12. Dependency and Supply Chain

## Unpinned Dependencies (Critical)

Every dependency in requirements.txt is unpinned:
- torch (installed: 2.2.2)
- torchaudio (installed, version unknown)
- numpy (unpinned)
- soundfile (unpinned)
- librosa (unpinned)
- sounddevice (unpinned)
- fastapi (unpinned)
- bigvgan (unpinned)
- campplus/funasr (unpinned)

## External Model Sources (No Hash Verification)

| Model | Source | Version Pin | Hash |
|---|---|---|---|
| DiT_uvit_tat_xlsr_ema.pth | huggingface.co/Plachta/Seed-VC | None | None |
| campplus_cn_common.bin | huggingface.co/funasr/campplus | None | None |
| rmvpe.pt | huggingface.co/lj1995/VoiceConversionWebUI | None | None |
| bigvgan_v2_22khz_80band_256x | huggingface.co/nvidia/bigvgan_v2 | None | None |

Any upstream update could silently break the application.

---

# 13. Connectivity Matrix — What Actually Connects

```
OFFLINE PATH (VERIFIED WORKING):
  run_offline_conversion.py
    -> SeedVCConverter
      -> _load_model() -> Seed-VC modules
      -> _run_inference() -> model.cfm.inference()
      -> soundfile.read/write

REST API PATH (PARTIAL):
  run_api.py / uvicorn
    -> FastAPI app
      -> /convert -> ConversionEngine -> SeedVCConverter
      -> /stream -> StreamingPipeline (NOT TESTED)
      -> /evaluate -> Evaluator
      -> /health -> OK
      -> /ws/stream -> WebSocketStreamHandler (NEW, UNTESTED with real audio)

STREAMING PATH (SCAFFOLDED, NOT VERIFIED):
  StreamingPipeline
    -> AudioCapture (exists, not tested in pipeline)
    -> RingBuffer (exists, not tested in pipeline)
    -> VAD (exists, not tested in pipeline)
    -> ConversionEngine (exists, not tested in pipeline)
    -> AudioPlayback (exists, not tested in pipeline)

WEBSOCKET PATH (NEW, UNVERIFIED):
  WebSocketStreamHandler
    -> converter.convert() per chunk
    -> ChunkAssembler for ordering
    -> No VAD integration
    -> No real audio tested
```

---

# 14. Current State Scorecard

| Area | Status | Confidence | Notes |
|---|---|---|---|
| audio capture | IMPLEMENTED/UNVERIFIED | MEDIUM | Code exists, never in live pipeline |
| buffer | IMPLEMENTED/UNVERIFIED | MEDIUM | Unit tests pass, never in live pipeline |
| playback | IMPLEMENTED/UNVERIFIED | MEDIUM | Code exists, never in live pipeline |
| VAD | PARTIAL | LOW | Module exists, Silero model never loaded in tests |
| Seed-VC integration | VERIFIED | HIGH | 5 successful conversions, model loads |
| model loading | VERIFIED | HIGH | 9.19s, 1241 MB, all components PASS |
| offline inference | VERIFIED | HIGH | 5 runs, output WAVs produced |
| accent transformation | UNVERIFIED | LOW | Output produced but accent shift not measured |
| speaker preservation | UNVERIFIED | LOW | Never measured |
| metrics | IMPLEMENTED/UNVERIFIED | MEDIUM | Code works, applied to real runs |
| RTF | VERIFIED | HIGH | Measured at 0.003-0.004 (includes load) |
| streaming | NOT IMPLEMENTED | HIGH | No chunked inference, no streaming |
| pipeline | PARTIAL | MEDIUM | Offline works, streaming unwired |
| stability | UNVERIFIED | LOW | 40 runs produced 5 successes, many failures |
| telephony readiness | NOT IMPLEMENTED | HIGH | 8kHz not supported anywhere |

---

# 15. Bugs and Risks (Prioritized)

| ID | Sev | File | Problem | Impact | Evidence | Fix |
|---|---|---|---|---|---|---|
| BUG-001 | P0 | seedvc.py:524 | .numpy() without .detach() | RuntimeError on some runs | run_027 traceback | Use .detach().numpy() |
| BUG-002 | P1 | seedvc.py:374 | RTF includes model load | Misleading performance claims | metrics.json shows 0.003 RTF | Separate load time from inference time |
| BUG-003 | P1 | capture.py | Unbounded queue | Memory growth under backpressure | Code inspection | Add max queue size + backpressure |
| BUG-004 | P1 | playback.py | Unbounded queue | Memory growth | Code inspection | Add max queue size |
| BUG-005 | P2 | seedvc.py | _load_model not idempotent | Double-load risk | Code inspection | Guard with _model check |
| BUG-006 | P2 | capture.py | Status flags silently dropped | Audio corruption undetected | Code inspection | Log/raise on overflow |
| BUG-007 | P2 | buffer.py | np.zeros per read_chunk | GC pressure | Code inspection | Pre-allocate or reuse |
| BUG-008 | P2 | capture.py | No device selection | Wrong microphone | Code inspection | Add device parameter |
| RISK-001 | P0 | seedvc.py | Model does accent conversion? | Product thesis invalid | Upstream analysis | Verify with human listening test |
| RISK-002 | P1 | seedvc.py | MPS never used | 5-10x slower inference | All 40 runs show device=cpu | Investigate MPS compatibility |
| RISK-003 | P1 | pipeline | Full-utterance only | Cannot do real-time | Code inspection | Requires model architecture change |
| RISK-004 | P2 | requirements.txt | No version pins | Reproducibility risk | File inspection | Pin all dependencies |
| RISK-005 | P2 | .git-backup/ | Leftover directory | Repository bloat | File system | Remove directory |

---

# 16. False or Premature Claims

The following claims should NOT be made based on current evidence:

1. "Real-time accent conversion" — NOT PROVEN. CPU inference is 3x slower than real-time. Streaming not implemented.
2. "Streaming support" — NOT PROVEN. WebSocket endpoints exist but never tested with real audio. StreamingPipeline is scaffolded.
3. "On-device processing" — PARTIALLY TRUE. Runs locally but on CPU, not MPS.
4. "Low latency" — FALSE. Wall-clock latency is 37-47 seconds per 12-second utterance on CPU.
5. "Speaker identity preserved" — UNVERIFIED. Never measured.
6. "Accent transformation" — UNVERIFIED. Output audio is produced but accent change not validated.
7. "P0 complete" — FALSE. P0 infrastructure exists but runtime validation incomplete.
8. "Production-ready" — FALSE. No error recovery, no monitoring, no auth, no rate limiting.

---

# 17. What Has Been Done Well

1. **Seed-VC reverse engineering** — Correctly identified that seed_vc is not a package, traced the actual API, matched config files to checkpoints.
2. **Persistence through failures** — 40 conversion runs with progressive debugging to reach working state.
3. **Security fix** — Path traversal in model_name was correctly patched.
4. **Resource management** — Model loading/unloading with RAM tracking.
5. **Metrics scaffolding** — RTF, latency, system monitoring infrastructure exists.
6. **Test coverage** — 216+ tests covering most modules (though heavily mocked).
7. **Temp file handling** — Uses tempfile with cleanup in finally block.
8. **Thread-safe buffer** — RingBuffer correctly uses locks for concurrent access.

---

# 18. What Must Not Be Changed Yet

1. **src/conversion/seedvc.py** — The _run_inference() method is the only known-working path to real conversion. Any changes risk breaking it.
2. **seed-vc/ submodule** — Contains the actual model code. Do not modify.
3. **MODEL_PRESETS** — The checkpoint/config pairing is verified working. Don't change.
4. **Temp file handling** — Despite latency concerns, it works. Optimize later.

---

# 19. Immediate Next Actions

## Priority 1: Validate Accent Shift (P0)

**Objective:** Confirm Seed-VC actually changes accent, not just voice timbre.
**Files:** src/conversion/seedvc.py, results/offline/run_040/output.wav
**Action:** Human listening test comparing source (Indian English) vs output vs reference (US English)
**Pass condition:** Listener confirms output sounds like Indian speaker with US pronunciation
**Failure interpretation:** If output sounds like US speaker, product thesis is invalid

## Priority 2: Test MPS Device (P0)

**Objective:** Determine why MPS is never used despite being available.
**Files:** src/conversion/seedvc.py, requirements.txt
**Action:** Run converter with device="mps" explicitly
**Pass condition:** Model loads and infers on MPS
**Failure interpretation:** Some model component doesn't support MPS; need CPU fallback strategy

## Priority 3: Measure True Inference RTF (P1)

**Objective:** Get accurate RTF excluding model load time.
**Files:** src/conversion/seedvc.py, scripts/run_offline_conversion.py
**Action:** Separate timing: load model once, then run 5 conversions
**Pass condition:** RTF < 1.0 on MPS or CUDA
**Failure interpretation:** Model is too slow for real-time; need smaller model or hardware acceleration

## Priority 4: Wire Streaming Pipeline (P1)

**Objective:** Connect AudioCapture -> RingBuffer -> VAD -> ConversionEngine -> AudioPlayback
**Files:** src/streaming/pipeline.py, src/audio/*.py, src/streaming/vad.py
**Action:** End-to-end test with real microphone and playback
**Pass condition:** Captured audio plays back through pipeline
**Failure interpretation:** Component incompatibilities, timing issues

## Priority 5: Pin Dependencies (P2)

**Objective:** Make environment reproducible.
**Files:** requirements.txt, seed-vc/requirements.txt
**Action:** Pin all versions based on currently installed packages
**Pass condition:** `pip install -r requirements.txt` produces identical environment
**Failure interpretation:** Dependency conflicts

---

# 20. Corrected Milestone Status

**CURRENT MILESTONE:** P0 infrastructure complete, P0 runtime validation incomplete

**COMPLETION:** 35%

**CORE THESIS PROVEN:** NO

**REAL MODEL LOAD PROVEN:** YES (model loads, all components initialize)

**REAL CONVERSION PROVEN:** YES (5 successful runs producing WAV output)

**ACCENT SHIFT PROVEN:** NO (never measured)

**SPEAKER PRESERVATION PROVEN:** NO (never measured)

**RTF MEASURED:** YES (but misleading — includes load time, CPU only)

**STREAMING PROVEN:** NO

**LIVE MIC PIPELINE PROVEN:** NO

**TELEPHONY READINESS:** NO (8kHz not supported)

**NEXT GATE:** Human listening test to validate accent shift. If output copies US speaker instead of shifting accent, product thesis requires pivot.

---

# 21. Final Engineering Judgment

1. **What exactly has been built?** An offline voice conversion system that takes 12-second audio files, runs them through Seed-VC on CPU, and produces converted WAV files. Infrastructure for REST API, WebSocket, VAD, streaming, and metrics exists but is not wired or verified.

2. **What has actually been proven to work?** Model loading (9s, 1241 MB RAM), full-utterance inference on CPU (37-47s for 12s audio), WAV output generation, RTF measurement, model unloading. 5 successful conversions from Indian English source to output using US English reference.

3. **What only appears to work from reading the code?** Streaming pipeline, WebSocket endpoints, VAD integration, live microphone pipeline, MPS acceleration, real-time operation. All have code but no runtime verification.

4. **What is still missing?** Accent shift validation, speaker preservation measurement, streaming architecture, MPS utilization, dependency pinning, backpressure handling, error recovery in streaming, telephony format support, authentication, rate limiting.

5. **What is currently the biggest technical risk?** Seed-VC is a voice converter, not an accent converter. The model may copy the reference speaker's voice rather than transform accent. This is an unproven assumption that could invalidate the product thesis. A human listening test of run_040/output.wav vs source vs reference would resolve this.

6. **Is Seed-VC actually suitable for the desired use case?** UNVERIFIED. Seed-VC is designed for zero-shot voice conversion (make source sound like reference). Accent conversion requires preserving source speaker identity while changing pronunciation. These are different problems. The tiny model may not have capacity for fine-grained accent modification without timbre change.

7. **Is the existing architecture suitable for continuing development?** YES, for offline development. The module separation is reasonable. For streaming development, the architecture needs significant changes: chunked inference, reference embedding caching, backpressure handling, and buffer management.

8. **Should anything be rewritten now?** NO. The core conversion path works. Fix the identified bugs (gradient leak, RTF calculation) and validate the product thesis before refactoring.

9. **What single experiment provides the most information next?** Human listening test of run_040/output.wav. Compare source (Indian English), reference (US English), and output. If output sounds like Indian speaker with US accent → proceed. If output sounds like US speaker → product thesis requires major pivot.

10. **What result would cause us to abandon Seed-VC?** If the output copies the reference speaker's voice instead of transforming the source speaker's accent. This would mean Seed-VC cannot solve the accent conversion problem.

11. **What result would justify moving into streaming development?** Confirmed accent shift with acceptable speaker preservation on single utterances, AND verified RTF < 0.5 on MPS or CUDA.

12. **How far is this project from a live call-centre-quality prototype?** Approximately:
- 1-2 weeks for accent validation (human listening test)
- 2-4 weeks for chunked inference (if model supports it)
- 4-8 weeks for streaming pipeline wiring and testing
- 2-4 weeks for latency optimization
- 2-4 weeks for stability testing
- **Total: 3-4 months minimum** for a basic prototype
- **6-12 months** for call-centre quality with interruptions, 8kHz support, and long-running stability
