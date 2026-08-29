# Training & Data Pipeline Analysis

## 1. Dataset Module (`src/accentedge/training/dataset.py`)

### Overview
`NativePriorDataset` is a config-driven PyTorch Dataset that yields:
- `zc1` — FACodec content codebook indices, shape `[1, T]` int64
- `phone_ids` — frame-level phoneme IDs at `codec_fps`, shape `[T]` int64
- `valid_frame_mask` — bool mask, shape `[T]`
- `speaker_id` — str
- `item_id` — str

### Audio Loading & Preprocessing
1. **Waveform loading**: `torchaudio.load(wav_path)` → tensor `[C, T]`.
2. **Resampling**: if `sr != sample_rate` (default 24k), uses `torchaudio.functional.resample`.
3. **Mono downmix**: averages multi-channel audio to single channel.
4. **Dtype**: cast to `torch.float32`.

### Latent Extraction (FAcodec Integration)
- Lazy-initialized `FACodecAdapter` (frozen) encodes waveform to `latents.content_zc1` → squeezed to `[T]` int64 codebook indices.
- `PhonemePipeline` maps transcript → frame-level `phone_ids` aligned to codec FPS.
- Both are **cached** under `cache_dir/<hh[:2]>/<hh>/`:
  - `latents.pt` — `zc1` with batch dim
  - `phone_ids.pt` — phone IDs with batch dim
  - `metadata.json` — speaker_id, item_id, num_frames, phone_frames

### Normalization
Optional per-channel z-score normalization of `zc1` when `normalize_latents: true`. Current default is `false`.

### Batching
`collate_fn` pads variable-length sequences to `T_max` in the batch:
- `zc1`: `[1, T_i]` → `[1, T_max]` → stack → `[B, 1, T_max]` int64
- `phone_ids`: `[T_i]` → `[T_max]` → stack → `[B, T_max]` int64
- `valid_mask`: padded with `False`
- speaker/item IDs returned as `List[str]`

### Config Keys
```yaml
dataset:
  sample_rate: 24000
  codec_hop: 300
  codec_fps: 80
  facodec_device: cpu
  normalize_latents: false
paths:
  facodec_checkpoint: Plachta/FAcodec
  cache_dir: ./cache/facodec_phones
```

---

## 2. Checkpoint System

### `src/accentedge/training/checkpoint.py` (Phase 1 Checkpoints)
**save_checkpoint** persists:
- `model_state_dict`
- `optimizer_state_dict`
- `scheduler_state_dict` (nullable)
- `step`, `epoch`
- `config_hash` (SHA-256 of canonical JSON config)
- `git_sha` (short HEAD, fallback `"unknown"`)
- `phone_vocab` (list[str])
- `facodec_ckpt` (HF identifier or path)
- `zc1_mean`, `zc1_std` — **required** per-channel normalization constants
- `config` (full dict)
- optional `extra`

**load_checkpoint** validates required keys and raises `ValueError` if normalization constants are missing.

### `src/accentedge/training/checkpoints.py` (Provenance Manifests)
Every checkpoint can have a sidecar `.json` manifest (`CheckpointManifest`) recording:
- checkpoint_id, architecture_id, version
- git_commit, config_hash, training_manifest_hash
- training_data_lineage_hash
- parent_checkpoint_ids, pretrained_weight_sources
- licenses, commercial_use_status
- seed, training_steps, training_hours
- optimizer, scheduler, hardware
- wall_clock_seconds, best_validation_metric
- timestamp

`save_checkpoint_manifest` derives a weights hash from the model state dict (pickle + SHA-256).

### FAcodec Integration
- The dataset caches **pre-extracted latents** so the frozen FACodec is only run once per file.
- Checkpoints store `facodec_ckpt` and normalization constants so inference can denormalize `zc1`.
- `overfit.py` loads `FACodecAdapter` and reuses its encode/decode for WAV reconstruction.

---

## 3. Overfit Test (`src/accentedge/training/overfit.py`)

### Purpose
Real-latent overfit training on 5–10 utterances to verify the denoiser learns before scaling to full training.

### Dataset
`OverfitDataset` loads `.wav` files from `audio_dir`, extracts `zc1`, `z_p`, `z_r`, `z_t` (approximated as zeros), and `zc2_target = z_q − zc1 − z_p − z_t − z_r` using `FACodecAdapter` and `PhonemePipeline`.

### Normalization
Per-channel zc1 mean/std computed over the entire overfit set; all tensors normalized before training.

### Loss
```python
loss = mse(eps_pred, noise) + zc2_loss_weight * mse(zc2_pred, zc2_n.detach())
```
`zc2_loss_weight` defaults to 0.5.

### Gates
Three mandatory gates evaluated every `gate_check_every` steps:

| Gate | What | Criterion |
|------|------|-----------|
| A — Denoising | Denoised zc1 closer to clean than noisy input | `avg_denoised < avg_noisy` AND improvement > 5% |
| B — Mean Baseline | Model beats predicting training-set mean | `model_loss < mean_baseline_loss` |
| C — Conditioning | Correct phones beat wrong + shuffled phones | `correct_loss < wrong_loss` AND `correct_loss < shuffled_loss` |

### WAV Decode
At specified steps and at the end, decodes one sample:
`denoised_zc1 + denoised_zc2 + z_p + z_r` → `FACodec.decode()` → WAV saved to `generated_wavs/`.

### Artifacts
- `checkpoint.pt` (latest), `checkpoint_step{N}.pt`
- `metrics.json`
- `zc1_stats.json`
- `generated_wavs/*.wav`

---

## 4. Trainer (`src/accentedge/training/trainer.py`)

### Design
Generic, device-aware, mixed-precision training loop with manifest sidecars.

### Loss Functions
Imported from `accentedge.models.training.losses`:
- `content_loss` — MSE
- `accent_loss` — MSE on accent embeddings
- `speaker_loss` — `1 - cosine_similarity` on speaker embeddings
- `f0_loss` — MSE on pitch contour
- `mel_loss` — L1 on mel spectrogram
- `reconstruction_loss` — L1 on waveform
- `total_loss` — weighted sum of present components

Default weights all `1.0`.

### Optimizer & Scheduler
Not instantiated inside `Trainer`; passed in externally. Utilities in `schedules.py`:
- Optimizers: Adam, AdamW, SGD via `get_optimizer`
- Schedulers: Linear, Cosine, Step, Constant via `get_lr_scheduler`

### Reproducibility
- `set_seed(seed)` sets Python, NumPy, PyTorch CPU/GPU RNGs.
- `enable_deterministic()` sets `cudnn.deterministic=True`, `benchmark=False`, and `use_deterministic_algorithms(True, warn_only=True)`.
- `get_rng_state()` captures Python hash seed, NumPy, and PyTorch CPU/CUDA RNG states.
- `verify_reproducibility()` runs model on dummy batch N times with same seed; asserts exact loss match.

### Mixed Precision
`fp16` autocast + `GradScaler` when `precision="fp16"` and device is CUDA. Otherwise `fp32`.

### Checkpointing
`save_checkpoint` writes:
- global_step, model_state, optimizer_state, scheduler_state (nullable), rng_state
- architecture_id, version

Then writes sidecar manifest via `save_checkpoint_manifest` with provenance fields.

### Fit Loop
```python
while global_step < max_steps:
    for batch in train_loader:
        train_step(batch)
        if val_loader and step % validation_every == 0:
            validate(val_loader)
            save best.pt if improved
        if step % checkpoint_every == 0:
            save step_{N}.pt
```
Final `final.pt` includes training hours and best validation metric.

---

## 5. Profiling Modules

### `src/accentedge/profiling/latency.py`
- `LatencyBreakdown` dataclass: frame_accumulation_ms, lookahead_ms, model_structural_ms, output_buffer_ms, compute_ms.
- `measure_chunk_latency` — wraps `candidate.process_chunk()` with `time.perf_counter()`, heuristic split of total compute_ms into sub-components (20/30/10/40).
- `algorithmic_latency_from_config` — computes latency purely from frame_ms, lookahead_ms, model_frames, buffer_frames.

### `src/accentedge/profiling/memory.py`
- `measure_model_memory` — sum of `param.numel() * element_size()` + buffers.
- `measure_session_memory` — delegates to `session.state_size_bytes()`.
- `profile_inference_memory` — returns dict with model/session/total bytes.

### `src/accentedge/profiling/rtf.py`
- `RTFMeasurement` dataclass with `rtf = compute_ms / audio_ms`.
- `measure_rtf` — iterates audio chunks, times `process_chunk()`, returns list of measurements.

### `src/accentedge/profiling/hardware.py`
- `HardwareProfile.capture()` — collects CPU model, cores, RAM (via `psutil` or `os.sysconf`), GPU model (torch.cuda), torch version, backend (cuda/mps/cpu), OS.

---

## 6. How Training Scripts Invoke These

### `scripts/train_phase1.py`
Smoke test for Colab:
- Instantiates tiny `DenoisingTransformerModel` (d_model=64, 2 layers).
- Generates random `zc1` and `phone_ids`; no real audio or FACodec.
- 50 training steps, asserts loss decreases.
- Does NOT use `Trainer`, `NativePriorDataset`, or checkpoint system.

### `scripts/gate1_manifest.py`
Environment manifest collector:
- Installs dependencies, clones FAcodec and accentedge.
- Records GPU name, Python/torch/CUDA versions.
- Hashes FAcodec checkpoint file.
- Saves `gate1_artifacts/environment.json`.

### `scripts/gate1_reconstruction.py`
Reconstruction equivalence test:
- Loads upstream FAcodec model and `FACodecAdapter`.
- Downloads 5 native (LibriSpeech) + 5 Indian English WAVs.
- Runs upstream encode→quantizer→decode and adapter encode→decode.
- Compares max abs diff (<1e-4), SNR (>60 dB), length.
- Saves WAVs and `reconstruction_metrics.json`.

### `scripts/gate2_identity.py`
Identity preservation calibration:
- Loads CMU ARCTIC (native) and L2-ARCTIC (Indian) via HuggingFace/torchaudio.
- Builds three-reference distributions: same-speaker, impostor, reconstruction.
- Uses ECAPA-TDNN (SpeechBrain) for embeddings.
- Computes `shift_over_span` and `preservation_ratio`; thresholds: shift < 0.25, preservation > 0.85.
- Saves `identity_calibration.json`.

### `scripts/gate4_strength_sweep.py`
Strength sweep for accent conversion:
- Loads L2-ARCTIC Indian English samples via `L2ArcticDataset`.
- Loads `FACodecAdapter` + `AccentConverter`; optionally loads denoiser checkpoint from `DENOISER_CKPT`.
- Runs conversion at strengths `[0.0, 0.25, 0.5, 0.75, 1.0]`.
- Evaluates mel L1, identity shift (ECAPA-TDNN), WER (faster-whisper).
- Gate 4 criteria: identity shift at s=0 < 0.02, at s=1 > 0.15, monotonic increase, mel L1 < 0.5.
- Saves `metrics_per_sample.json`, `strength_curves.json`, `gate4_manifest.json`, plot PNG.

---

## 7. What's Missing for Real Training

| Category | Gap | Detail |
|----------|-----|--------|
| **Data Loading** | No streaming / WebDataset support | `NativePriorDataset` requires all items enumerated up front; no sharding for multi-node. |
| | No train/val split logic | Scripts load ad-hoc subsets; `TrainingManifest` has split field but dataset doesn't consume manifests. |
| | No on-the-fly augmentation | No noise, speed perturbation, pitch shift, or SpecAugment. |
| | Single-speaker / utterance caching | Cache key is `sha256(item_id)[:16]`; collisions possible across runs with different speakers. |
| **Preprocessing** | Silence / VAD trimming | `valid_frame_mask` is all-ones; no energy-based masking. |
| | Phoneme alignment robustness | `PhonemePipeline` output assumed to match codec frames exactly; no duration normalization. |
| | Text normalization | No script for cleaning transcripts (numbers, symbols, casing). |
| **Training Loop** | No distributed training | `Trainer` assumes single device; no DDP/FSDP, no `DistributedSampler`. |
| | No gradient accumulation | Fixed batch-size training; no accumulation for large models. |
| | No checkpoint rotation | Saves all checkpoints indefinitely; no keep-last-N policy. |
| | No early stopping | Only best-val checkpointing; no patience-based halt. |
| | No TensorBoard / WandB | Logger is a callable defaulting to print; no structured experiment tracking. |
| **Reproducibility** | Seed not fully deterministic | `torch.use_deterministic_algorithms` not enforced in `Trainer`; `enable_deterministic()` exists but is never called in the training path. |
| | No deterministic DataLoader | `worker_init_fn` and `generator` not set for PyTorch workers. |
| | RNG state not fully restored | `load_checkpoint` restores model/optimizer/scheduler but not RNG seeds. |
| **Model** | No multi-GPU | `FACodecAdapter` loaded on single device; no model parallelism. |
| | `z_t` approximated as zeros | `overfit.py` comment admits FACodecAdapter doesn't expose timbre residual; real training needs full factorized latent extraction. |
| **Evaluation** | No validation set pipeline | `Trainer.validate` exists but no dataset/loader wiring; overfit test only evaluates gates on training data. |
| | No held-out test set | No final evaluation protocol or metric reporting (BLEU, SECS, etc.). |
| **Lineage & Compliance** | Manifest not consumed | `TrainingManifest` and `DataLineage` exist but no training script reads them. |
| | License check missing | `validate_training_manifest` checks speaker overlap but not commercial-use propagation before training. |
| **Infrastructure** | No CI/CD for gates | Gates are run manually; no automated gate progression on PR. |
| | No Docker / container definition | Hardware-specific paths (e.g., `/content/FAcodec`) baked into scripts. |
| | No hyperparameter sweep | `gate4_strength_sweep.py` is inference-only; no training sweep. |

### Summary
The repository has solid scaffolding for data schemas, provenance manifests, profiling, and overfit validation. However, **real training** at scale requires: distributed data loading, deterministic DataWorkers, manifest-driven dataset splitting, gradient accumulation, proper RNG restoration, full factorized latent extraction from FACodec, and automated gate execution in a reproducible CI environment.
