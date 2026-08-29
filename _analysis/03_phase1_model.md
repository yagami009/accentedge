# Phase 1 Model Analysis — FAC-FACodec

_Generated from source audit of `src/accentedge/phase1/`, `src/accentedge/codec/`, and `docs/phase1/`._

---

## 1. The FAC-FACodec Paper's Core Idea

**Paper**: `arxiv:2510.10785v2` — *FAC-FACodec: Factorized Accent Control for Speech Codecs*

### 1.1 Central Thesis

The paper argues that neural speech codecs (FAcodec, DAC, EnCodec) entangle **accent/phonetic content** with **prosody, timbre, and acoustic detail** into a single quantized latent. To convert accent without changing speaker identity or prosody, you must **factorize** that latent into independently-controllable components, modify only the accent-bearing component, and recompute the parts that depend on it.

### 1.2 The Factorization

The paper's actual factorization (verified from both the paper and the upstream FAcodec quantizer code) is:

```
z = encoder(wav)
content_quantizer  (2 codebooks)  → z_c1 + z_c2
prosody_quantizer  (1 codebook)   → z_p
residual_quantizer (3 codebooks)  → z_r
timbre_quantizer   (2 codebooks)  → z_t  [timbre_norm=True: this path is replaced by StyleEncoder]

combined = z_p + z_c + z_r  (timbre modulation via gamma/beta LayerNorm)
decoder(combined) → waveform
```

Key insight: **content is split into TWO residuals** — `z_c1` (primary phonetic content, 8-dim, first RVQ codebook) and `z_c2` (secondary/finer content detail, 8-dim, second RVQ codebook). Their sum `z_c = z_c1 + z_c2` is what the codec normally returns.

**`z_c2` is NOT preserved during accent conversion.** It must be **recomputed** from the modified `z_c1` and the encoder features. This is the paper's critical design constraint.

### 1.3 The Diffusion Mechanism

- The diffusion model operates on **`z_c1` only** (the first content codebook output).
- Noise schedule: linear `β_t ∈ [10⁻⁴, 2×10⁻²]`, `T = 100` timesteps.
- Loss: `‖ε - s_θ(x_t, t, π)‖₂²` where ε is predicted noise.
- Conditioning: phoneme embeddings `π` via **FiLM** (affine modulation) + additive timestep embeddings.
- Architecture: 6-layer Transformer encoder, 8 attention heads, `d_model=1024`, FFN `2048`, dropout `0.1`.
- Inference: **DDIM** ODE formulation with `t_start` controlling partial denoising strength.

### 1.4 Inference Pipeline

```
1. Encode source: z_c1s, z_p, z_r, g, encoder_features
2. Get phonemes π for source transcript (eSpeak-ng + Wav2Vec2-XLSR alignment)
3. Add noise to z_c1s up to t_start
4. Denoise: ŷ_c1 = denoiser(noisy_z_c1, π, t_start)
5. Interpolate: z_c1_new = (1-strength) * z_c1_orig + strength * ŷ_c1
6. Predict z_c2_new = fc_zc2_head(transformer_h, ŷ_c1)
7. Recombine: z_combined = z_p + (z_c1_new + z_c2_new) + z_r
8. Timbre modulate: z_q = gamma * LayerNorm(z_combined) + beta
9. Decode: waveform = decoder(z_q)
```

### 1.5 Frame Rate

The paper uses **20 ms frames (50 fps)** per PAPER_VERIFICATION.md. However, the AccentEdge code uses **80 fps** (hop_length=300, sr=24000 → 24000/300=80), which is verified from the FACodec config. This is a **discrepancy** — the code does not match the paper's reported 50 fps.

---

## 2. Codec Adapter Interface (`interfaces.py` + `facodec.py`)

### 2.1 `interfaces.py` — The Protocol

```python
@dataclass
class FactorizedLatents:
    content: torch.Tensor          # [B, C, T] — quantized content (z_q, timbre-conditioned)
    content_zc1: torch.Tensor      # [B, 1, T] — first content codebook indices
    content_zc2: Optional[torch.Tensor]  # [B, C, T] — second content codebook indices
    prosody: Optional[torch.Tensor]      # [B, 1, T] — prosody codebook indices
    detail: Optional[torch.Tensor]       # [B, K, T] — residual detail codebook indices
    timbre: Optional[torch.Tensor]       # [B, D] — global timbre embedding
    metadata: dict

class FactorizedSpeechCodec:
    sample_rate: int
    def encode(self, waveform) -> FactorizedLatents: ...
    def decode(self, latents) -> torch.Tensor: ...
    def freeze(self) -> None: ...
    def parameters(self): ...
```

This is a clean protocol design: any codec can implement the interface, and AccentEdge doesn't couple to a specific upstream implementation.

### 2.2 `FACodecAdapter` — How It Wraps Plachta/FAcodec

The adapter wraps the upstream FAcodec (Plachta/FAcodec on HuggingFace Hub) with this flow:

```
encoder(wav) → z
quantizer(z, wav, n_c=2) → z_q, [z_p, z_c, z_r], losses, timbre  (timbre_norm=True)
                              OR [z_c, z_p, z_t, z_r]              (timbre_norm=False)
decoder(z_q) → waveform
```

**Critical design decisions:**

1. **Timbre is baked into z_q**: When `timbre_norm=True` (default), the quantizer applies StyleEncoder → gamma/beta → LayerNorm modulation directly on the summed latent. `z_q` returned by the quantizer already has timbre baked in. The decoder expects exactly this `z_q`.

2. **Individual codebook exposure via `content_all_quant`**: The `forward_v2` method returns a 6th element — `content_all_quant` — which is a list of individual codebook outputs: `[zc1, zc2]`. The adapter extracts these and assigns them to `content_zc1` and `content_zc2`. This is a **key fix** documented in ZC2_CONTRACT.md as the correct approach.

3. **`content_zc1` and `content_zc2` shapes**: Both are `[B, 1, T]` in the adapter's return. The ZC2 contract says these should be 8-dim (the codebook dimension after `out_proj`), but the adapter returns them as `[B, 1, T]` (the codebook INDEX dimension). This is because `content_all_quant[i]` from the upstream returns the projected codebook output, not the indices.

4. **Freeze**: All parameters are frozen after loading, enforced with `requires_grad = False` and an assertion in `freeze()`.

### 2.3 Bug Fixed (Relative to ZC2_CONTRACT Analysis)

The ZC2_CONTRACT.md §4 documents that an earlier version of the adapter assigned the **full `z_c` sum** to `content_zc1` and left `content_zc2=None`. The current code (lines 141-143) correctly extracts `zc1 = content_all_quant[0]` and `zc2 = content_all_quant[1]` from the 6th return value of `forward_v2`. The contract doc is a historical record of a resolved bug.

---

## 3. Diffusion Model (`diffusion.py`)

### 3.1 Noise Schedule

```python
def compute_noise_schedule(num_steps=100, noise_min=1e-4, noise_max=2e-2):
    betas = torch.linspace(noise_min, noise_max, num_steps)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {"betas": ..., "alphas": ..., "alpha_bar": ...,
            "sqrt_alpha_bar": ..., "sqrt_1m_alpha_bar": ...}
```

- Linear schedule matching the paper's `β_t ∈ [10⁻⁴, 2×10⁻²]`.
- `sqrt_alpha_bar[t]` and `sqrt_1m_alpha_bar[t]` are precomputed for efficient sampling.
- Note: `diffusion.py` uses `noise_max=2e-2` while `denoiser.py` internally uses `0.02` (same value, hardcoded). The converter reads buffers from the denoiser, not from `diffusion.py`, so the schedule is consistent in practice.

### 3.2 Forward Diffusion (`q_sample`)

```python
def q_sample(x0, t, sqrt_alpha_bar, sqrt_1m_alpha_bar):
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar[t].view(-1,1,1) * x0 + sqrt_1m_alpha_bar[t].view(-1,1,1) * noise
    return xt, noise
```

Standard DDPM forward process: `x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε`.

### 3.3 DDIM Step

```python
def ddim_step(x_t, eps_pred, t, t_prev, sqrt_alpha_bar, sqrt_1m_alpha_bar, eta=0.0):
    x0_pred = (x_t - sqrt_1ma_t * eps_pred) / sqrt_a_t
    x0_pred = torch.clamp(x0_pred, -10, 10)
    if t_prev is None or t_prev < 0:
        return x0_pred  # terminal step: return x0 estimate
    x_t = sqrt_a_prev * x0_pred + sqrt_1ma_prev * eps_pred + eta * sqrt_1ma_prev * noise
    return x_t
```

- `eta=0.0` gives the deterministic DDIM ODE path (matching the paper's "DDIM ODE formulation").
- `eta>0` would give stochastic DDPM sampling.
- The function is defined but **not called by the converter** — the converter implements its own simpler DDPM-style inference in `_run_denoiser()`.

### 3.4 Strength Control

```python
def strength_to_t_start(strength: float, num_steps: int = 100) -> int:
    return int(round(max(0.0, min(1.0, strength)) * num_steps))
```

Linear mapping: `strength=0.0 → t_start=0` (no denoising), `strength=1.0 → t_start=100` (full denoising). This is a **simplified implementation** — the paper may use a different `t_start` mapping strategy.

**Note**: `strength.py` and `denoiser.py` each define their own `StrengthScheduler` class (duplication). The converter doesn't use either directly — it computes `t_start` inline in `_run_denoiser()`.

---

## 4. Denoiser (`denoiser.py`)

### 4.1 Architecture

**`DenoisingTransformerModel`** — the core paper-faithful reimplementation:

```
Input: zc1_noisy [B, C=8, T]  +  phone_ids [B, T]  +  timestep t [B]
  │
  ├─ input_proj: Linear(8 → 1024)  →  h [B, T, 1024]
  │
  ├─ phone_emb: Embedding(393, 1024)  →  phone_emb [B, T, 1024]
  ├─ phone_proj: Linear(1024 → 2048)  →  phone_cond [B, T, 2048]
  │
  ├─ t_embed: SinusoidalPosEmb(1024) → Linear → SiLU → Linear  →  t_emb [B, 2048]
  │   t_emb expanded to [B, T, 2048]
  │
  ├─ cond = phone_cond + t_emb  [B, T, 2048]
  │
  ├─ CustomTransformerEncoder (6 layers):
  │   Each layer:
  │     MultiheadAttention(8 heads, 1024 dim)
  │     LayerNorm
  │     ConvFeedForward: Conv1d(1024→2048) → ReLU → Dropout → Conv1d(2048→1024) → Dropout
  │     CondLayerNorm (FiLM: gamma, beta from cond)
  │
  ├─ eps_pred = fc_out(h)  [B, T, 1024] → [B, 8, T]
  │
  └─ zc2_pred:
        x0_hat = (zc1_noisy - sqrt(1-abar[t])*eps_pred.detach()) / sqrt(abar[t])
        zc2_input = cat([h, x0_hat.detach()], dim=-1)  [B, T, 1024+8]
        zc2_pred = fc_zc2(zc2_input)  Linear(1032→4096) → GELU → Linear(4096→8)
                      → [B, 8, T]
```

### 4.2 Paper Fidelity Assessment

| Paper Spec | Implementation | Status |
|---|---|---|
| 6-layer Transformer | `num_layers=6` default | ✅ Match |
| 8 attention heads | `nhead=8` default | ✅ Match |
| d_model=1024 | `d_model=1024` default | ✅ Match |
| FFN=2048 | `d_ff=2048` default | ✅ Match |
| Dropout=0.1 | `dropout=0.1` default | ✅ Match |
| Sinusoidal timestep emb | `SinusoidalPosEmb` + MLP | ✅ Match |
| Phoneme conditioning via FiLM | `CondLayerNorm` with phone_proj | ✅ Match |
| Epsilon prediction | `fc_out` head | ✅ Match |
| zc2 prediction head | `fc_zc2` (h + x0_hat → zc2) | ✅ Partial (see below) |
| DDIM inference | Defined in `diffusion.py` but **not used** by converter | ⚠️ Divergence |

**`fc_zc2` fidelity**: The paper says "z_c2 is predicted from the denoised z_c1 and encoder features." The implementation concatenates transformer hidden states `h` (which encode encoder features via input projection + attention) with `x0_hat` (the denoised z_c1 estimate). This is a reasonable interpretation but not guaranteed to match the official implementation (which is not publicly released).

### 4.3 Tests (11 tests in `test_phase1.py` + integration tests)

| Test Class | Tests | What's Covered |
|---|---|---|
| `TestDiffusion` | 3 | Schedule shape, q_sample shape, DDIM shape |
| `TestStrength` | 2 | strength↔t_start mapping, StrengthScheduler |
| `TestDenoiser` | 6 | SinusoidalPosEmb, CondLayerNorm, ConvFeedForward, transformer layer, encoder, full forward pass |
| `TestCodecInterface` | 1 | FactorizedLatents construction |

Integration tests in `test_converter_pipeline.py` add:
- Full pipeline with mocked codec: shape preservation, intermediate keys, strength=0 identity, strength variance
- ZC2Recomputer: predict mode returns correct types/shapes, no gradients, missing phone_ids raises
- Frame rate contract: 80fps consistency across durations
- Error propagation: broken adapter, broken phoneme pipeline, invalid wav shape
- `freeze()` sets `requires_grad=False`

**Test quality**: Good shape/contract coverage. Missing: numerical correctness tests (e.g., ddpm loss values, zc2 prediction accuracy against ground truth), gradient flow tests, multi-batch tests, and DDIM path validation.

---

## 5. Phoneme Pipeline (`phoneme_pipeline.py`)

### 5.1 Pipeline Flow

```
transcript (text)
  → eSpeak-ng via phonemizer → IPA phoneme sequence
  → Wav2Vec2-XLSR-53 CTC forced alignment → frame-level phone boundaries (seconds)
  → 80fps frame mapping → [1, T] phoneme ID tensor
```

### 5.2 Frame Rate

- **80 fps** (hop_length=300, sr=24000): `num_frames = int(round(num_samples * 80 / 24000))`
- This matches the FACodec frame rate derived from its config.
- **Note**: The paper reports 20 ms frames (50 fps). The code uses 80 fps. This discrepancy is noted but not explained.

### 5.3 Alignment Algorithm

The `_viterbi_align()` function is a **simplified** implementation:

1. Greedy CTC decode: take `argmax` per frame to get frame-level labels
2. Merge consecutive same labels into decoded sequence
3. Proportional assignment: each target phone gets `n_ctc / n_phones` frames
4. Refinement: snap boundaries to decoded label-change positions

**Assessment**: This is not a true Viterbi/forced alignment. It's a greedy + proportional heuristic. It will produce reasonable results for clean speech with accurate transcripts but will fail on:
- Misaligned transcripts (wrong words → wrong phone sequence)
- Fast speech where phones overlap
- Silence/pause handling (currently skipped entirely — `sp` is filtered out)

### 5.4 Phoneme Vocabulary

- 93 IPA symbols covering English monophthongs, diphthongs, consonants, stress markers.
- Vocabulary IDs 0..92 (well within the denoiser's embedding table of 393 rows).
- `PAD_ID=1` (`sp` = silence/space) — distinct from the PyTorch embedding `padding_idx=392`.
- The vocabulary contract is tested: 393 denoiser rows, 93 active symbols, no ID collisions.

### 5.5 Dependencies

- `phonemizer` + `eSpeak-ng` (system binary) — required for text→phones
- `transformers` + `Wav2Vec2ForCTC` — required for alignment
- `torchaudio` — required for 16kHz resampling

All are lazily loaded with clear error messages.

---

## 6. Converter (`converter.py`)

### 6.1 Full Pipeline Assembly

```python
class AccentConverter:
    def convert(wav, transcript, strength=1.0) -> waveform:
        # Step 1: Encode
        latents = facodec_adapter.encode(wav)
        z_q, z_c1, z_p, z_r, g = extract(latents)

        # Step 2: Phoneme conditioning
        phone_ids = phoneme_pipeline(transcript, wav)  # [1, T_frames]

        # Step 3: Denoise
        z_q_denoised, zc2_pred = _run_denoiser(z_q, phone_ids, strength)

        # Step 4: ZC2 recomputation
        zc2_result = zc2_recomputer.recompute(
            encoder_features=z_q,
            modified_zc1=z_q_denoised,
            z_p=z_p, z_r=z_r,
            phone_ids=phone_ids
        )

        # Step 5: Decode
        output_latents = latents
        output_latents.content = z_q_denoised
        output_latents.content_zc2 = zc2_result.zc2
        output_wav = facodec_adapter.decode(output_latents)
```

### 6.2 Denoiser Execution (DDPM-style)

```python
def _run_denoiser(z_q, phone_ids, strength):
    t_start = int(round(strength * num_steps))
    if t_start == 0:
        # Still run denoiser at t=0 to get zc2 prediction
        eps_pred, zc2_pred = denoiser(z_q, phone_ids, t=0)
        return z_q, zc2_pred

    # Add noise
    noise = torch.randn_like(z_q)
    z_noisy = sqrt_abar[t_start] * z_q + sqrt_1m_abar[t_start] * noise

    # Run denoiser
    eps_pred, zc2_pred = denoiser(z_noisy, phone_ids, t_start)

    # Recover x0
    x0_hat = (z_noisy - sqrt_1m_abar[t_start] * eps_pred) / sqrt_abar[t_start]

    # Interpolate
    z_q_denoised = (1 - strength) * z_q + strength * x0_hat
    return z_q_denoised, zc2_pred
```

**Important**: The denoiser operates on `z_q` (the full 8-dim quantized content, timbre-conditioned), not on `z_c1` alone. The docstring says "the denoiser operates on the full 8-dim z_q representation, NOT on the 1-dim z_c1." This is **different from the paper**, which operates on `z_c1` (the first codebook output). The converter then passes `z_q_denoised` (8-dim continuous) to the ZC2Recomputer as `modified_zc1`.

### 6.3 Validation

- Input shape validation (`_validate_latent_shapes`): checks batch size and frame count consistency across `z_q`, `z_c1`, `z_p`, `z_r`, `g`.
- Frame rate contract (`_assert_frame_rate_contract`): `phone_ids.shape[-1]` must equal `z_q.shape[-1]`.
- Strength clamping: `strength` clamped to `[0, 1]`.

---

## 7. Strength Control (`strength.py`) and ZC2 Recompute (`zc2_recompute.py`)

### 7.1 Strength Control

```python
def strength_to_t_start(strength, num_steps=100) -> int:
    return int(round(clamp(strength, 0, 1) * num_steps))

class StrengthScheduler:
    def __call__(strength) -> int: ...
    def validate(strength) -> bool: ...
    def available_strengths() -> list:  # [0.0, 0.25, 0.50, 0.75, 1.0]
```

**Status**: Functional but simplified. Linear mapping from `[0,1]` to `[0, num_steps]`. The converter computes `t_start` inline rather than using this module. The `StrengthScheduler` class is duplicated in `denoiser.py` as well.

### 7.2 ZC2 Recompute

```python
class ZC2Recomputer:
    mode: 'predict' | 'recompute'

    def recompute(encoder_features, modified_zc1, z_p, z_r, phone_ids) -> ZC2RecomputeResult:
        if mode == 'predict':
            # Run denoiser at t=0, get fc_zc2 head output
            _, zc2_pred = denoiser(modified_zc1, phone_ids, t=0)
            return ZC2RecomputeResult(zc1=modified_zc1, zc2=zc2_pred, ...)
        else:
            # Re-run second codebook on residual
            residual = encoder_features - z_p.detach() - modified_zc1
            zc2 = quantizer.content_quantizer.quantizers[1](residual)
            return ZC2RecomputeResult(zc1=modified_zc1, zc2=zc2, ...)
```

**Two modes**:

1. **`predict` (default, paper-faithful)**: Runs the denoiser at `t=0` to get `fc_zc2` head output. No noise is added — `x0_hat = modified_zc1` at `t=0`. The transformer uses the denoised `zc1` as input and phoneme conditioning to predict `zc2`.

2. **`recompute`**: Attempts to re-run the second codebook of the content RVQ on the residual. Has three fallback strategies:
   - Direct access: `quantizer.content_quantizer.quantizers[1]`
   - RVQ with `start_codebook=1`
   - Straight-through VQ estimator (trainable placeholder — warns user)

**`build_content_latent()`**: Static utility combining factors: `z_p + (zc1 + zc2) + z_r`. Note: this doesn't include timbre modulation — the adapter's `decode()` expects `z_q` which already has timbre baked in. The converter doesn't call `build_content_latent()` directly; it sets `output_latents.content` and `output_latents.content_zc2` and calls `decode()`.

---

## 8. What's Working vs. What's Scaffolded

### ✅ Working (Functional Code with Tests)

| Component | Status | Evidence |
|---|---|---|
| Codec interface (`FactorizedLatents`, `FactorizedSpeechCodec`) | ✅ Working | 1 test + integration tests; clean protocol |
| FACodec adapter (encode/decode) | ✅ Working | Mocked in integration tests; `freeze()` tested |
| Denoiser architecture | ✅ Working | 6 unit tests; 11-total test suite passes |
| Diffusion math (schedule, q_sample, DDIM) | ✅ Working | 3 unit tests |
| Strength mapping | ✅ Working | 2 unit tests |
| Phoneme pipeline (phones_to_frames, frame count contract) | ✅ Working | Extensive unit tests (mocked alignment) |
| ZC2 recompute (predict mode) | ✅ Working | 4 unit + 3 integration tests |
| Full converter pipeline (mocked) | ✅ Working | 11 integration tests |
| Frame rate contract (80fps) | ✅ Working | Parametrized across durations |
| Vocabulary contract (393 denoiser rows, 93 active symbols) | ✅ Working | 6 dedicated tests |

### ⚠️ Partially Working / Needs GPU Verification

| Component | Status | Issue |
|---|---|---|
| FACodec reconstruction (encode→decode round-trip) | 🔄 Ready, untested | Needs Colab T4; mel L1 gate <0.15 |
| Real phoneme alignment (Wav2Vec2-XLSR) | 🔄 Ready, untested | Needs Colab for GPU inference |
| Strength sweep (0/0.25/0.5/0.75/1.0) | 🔄 Ready, untested | Listed in IMPLEMENTATION.md next steps |
| Actual accent conversion quality | 🔄 Not yet demonstrated | No end-to-end audio evaluation yet |

### 🚫 Known Bugs / Design Issues

| Issue | Severity | Location | Notes |
|---|---|---|---|
| **Frame rate mismatch (80fps vs paper's 50fps)** | Medium | `phoneme_pipeline.py:427`, `converter.py` | Code uses 80fps (FACodec hop=300); paper says 20ms (50fps). May cause temporal misalignment if paper-trained denoiser expects 50fps. |
| **DDIM defined but unused** | Low | `diffusion.py:24-39` | DDIM step function exists but `AccentConverter._run_denoiser()` uses simpler DDPM-style inference, not DDIM. |
| **Duplicate `StrengthScheduler`** | Low | `denoiser.py:15-30` and `strength.py:24-39` | Two identical classes. Converter uses inline computation. |
| **Duplicate `strength_to_t_start`** | Low | `diffusion.py:42-44` and `strength.py:13-16` | Two identical functions. |
| **`zc2_recompute.py` `_align_channels` creates unfrozen nn.Linear** | Low | `zc2_recompute.py:355-359` | Creates a `nn.Linear` on-the-fly without registering it; will not appear in `model.parameters()`. Cosmetic for inference. |
| **Viterbi alignment is heuristic, not true Viterbi** | Medium | `phoneme_pipeline.py:242-377` | Uses greedy CTC + proportional assignment, not proper Viterbi DP. May misalign phones on challenging audio. |
| **Denoiser operates on `z_q` (8-dim continuous), not `z_c1` (1-dim indices)** | High (design choice) | `converter.py:164` | The paper operates on `z_c1` codebook output. The code operates on the full `z_q` latent. This changes what the diffusion model is learning to denoise. |
| **`build_content_latent()` doesn't include timbre modulation** | Medium | `zc2_recompute.py:366-390` | The combined `z_p + (zc1+zc2) + z_r` lacks the StyleEncoder gamma/beta modulation. The adapter's `decode()` expects `z_q` that already has timbre baked in. The converter sidesteps this by setting `output_latents.content = z_q_denoised` (passing through the denoised full latent), but the `content_zc2` field isn't used by `decode()`. |

### 🚧 Scaffolded / Not Yet Implemented

| Component | Status | Notes |
|---|---|---|
| **Training scripts** (`scripts/train_phase1.py`) | 🚧 Scaffolded | Listed in IMPLEMENTATION.md but not audited in source |
| **Evaluation metrics** (`evaluation/`) | 🚧 Partial | Referenced in IMPLEMENTATION.md; files exist in `tests/` but not audited here |
| **Colab bootstrap** (`scripts/colab_bootstrap.sh`) | 🚧 Scaffolded | Listed; not audited |
| **Reconstruction verification** (`verify_facodec_direct.py`) | 🟡 Ready | Needs GPU |
| **Target accent extraction** | 🚧 Not implemented | The converter only handles source-side denoising. The paper's pipeline also encodes a target-accent reference to extract `z_c1_target`. This is not yet in the code. |
| **Timbre transfer strategy** | 🚧 Not implemented | Whether timbre comes from source or target is marked "TBD per strategy" in the adapter docstring |
| **Actual training** | 🚧 Not started | No training run has been executed |

---

## 9. Critical Architectural Concerns

### 9.1 The `z_q` vs `z_c1` Mismatch

The converter operates the denoiser on `z_q` (the full 8-dim timbre-conditioned continuous latent) rather than on `z_c1` (the first codebook's 8-dim output). The paper's diagram shows the diffusion operating on `z_c1` specifically. If `z_q ≠ z_c1` (which it isn't — `z_q` includes `z_p + z_c + z_r` with timbre modulation), then the denoiser is learning to denoise a different representation than the paper describes. This is the highest-severity design concern.

### 9.2 The Decoder Path Issue

The adapter's `decode()` method takes `latents.content` (which the converter sets to `z_q_denoised`, shape `[B, 8, T]`) and passes it directly to `decoder()`. The decoder expects `[B, 1024, T]`. If `z_q_denoised` is only 8-dim, the decoder will fail or produce garbage. The adapter's encode path returns `z_q` which is `[B, 8, T]` (extracted from `result[0]` of `forward_v2`), but the actual FACodec decoder expects 1024-dim input. **This is a potential runtime bug** that only manifests when a real FAcodec model is loaded (the mock stubs don't catch shape mismatches).

### 9.3 Missing `z_c` Reconstruction Path

The adapter docstring says the accent conversion combine is `z_modified = z_c_target + z_p + z_r`, but the actual implementation in the converter sets `output_latents.content = z_q_denoised` (the full denoised z_q) and `output_latents.content_zc2 = zc2`. The `decode()` method only reads `latents.content` — it completely ignores `content_zc2`. So the zc2 recomputation has **no effect on the output** with the current decode path.

### 9.4 Data Flow Summary

```
encode() returns FactorizedLatents where:
  content = z_q [B, 8, T]      ← THIS is what decode() uses
  content_zc1 = zc1 [B, 1, T]  ← never used by decode()
  content_zc2 = zc2 [B, 1, T]  ← never used by decode()

converter sets:
  output_latents.content = z_q_denoised  ← used by decode()
  output_latents.content_zc2 = zc2       ← NOT used by decode()

decode():
  z_q = latents.content  →  decoder(z_q)  → waveform
```

The `zc2_recomputer` and the entire zc2 machinery is **dead code** in the current pipeline because `decode()` ignores `content_zc2`. To fix this, `decode()` must reconstruct `z_q = z_p + (zc1 + zc2) + z_r` with timbre modulation before passing to the decoder.

---

## 10. Summary

The Phase 1 model implements the FAC-FACodec paper's core idea — factorized latents with diffusion-based accent conversion — with paper-faithful denoiser architecture and a working mock-tested pipeline. The codec interface is clean and extensible. The phoneme pipeline has solid frame-rate contract testing. The diffusion math is correct and tested.

However, there are three **critical gaps** that prevent the pipeline from working end-to-end with real models:

1. **The decode path ignores `content_zc2`** — the zc2 recompute has no effect on output audio.
2. **The denoiser operates on `z_q` (combined latent) instead of `z_c1` (first codebook only)** — deviates from the paper's design.
3. **The FACodec decoder expects 1024-dim input but the adapter passes 8-dim `z_q`** — will crash with a real model.

These are all fixable — the ZC2_CONTRACT.md already documents the required reconstruction formula (`z_q = z_p + (zc1 + zc2) + z_r` with timbre modulation) — but the implementation has not been applied to `decode()` or the converter's final assembly step.
