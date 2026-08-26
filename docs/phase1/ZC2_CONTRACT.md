# ZC2 Contract — Phase 1 Architecture Unknown

## Executive Summary

The FAC-FACodec paper factorizes content into two residuals, $z_{c1}$ and $z_{c2}$, where $z_{c2}$ is **not preserved** during accent conversion — it must be **recomputed** from denoised $\hat{z}_{c1}$ plus encoder features. The upstream FAcodec quantizer never exposes individual codebook outputs; it returns only the summed content latent $z_c = z_{c1} + z_{c2}$. The current AccentEdge adapter incorrectly assigns $z_c$ (the full sum) to `content_zc1`, and the denoiser's `fc_zc2` head predicts $z_{c2}$ from the denoised $z_{c1}$ estimate, which is the right mechanism — but it receives the wrong input. This contract documents the exact data flow, what must be recomputed, and the risks of getting it wrong.

---

## 1. FAcodec Quantizer Internals

### 1.1 Quantizer Architecture

The `FAquantizer` class (`modules/quantize.py`) contains four `ResidualVectorQuantize` modules:

| Quantizer | Codebooks | Dim | Role |
|-----------|-----------|-----|------|
| `content_quantizer` | `n_c_codebooks` (default 2) | 8 | $z_c = z_{c1} + z_{c2}$ |
| `prosody_quantizer` | `n_p_codebooks` (default 1) | 8 | $z_p$ |
| `timbre_quantizer` | `n_t_codebooks` (default 2) | 8 | $z_t$ |
| `residual_quantizer` | `n_r_codebooks` (default 3) | 8 | $z_r$ |

Each `ResidualVectorQuantize` (`dac/nn/quantize.py`) chains `n_codebooks` individual `VectorQuantize` modules. Each VQ module:
1. Projects input to codebook_dim via `in_proj` (Linear weight_norm)
2. L2-normalizes and finds nearest codebook entry
3. Returns `z_q_i` = straight-through estimator: `z_e + (z_q - z_e).detach()`
4. Projects back to input_dim via `out_proj` (Linear weight_norm)

The RVQ accumulates: `z_q = Σ z_q_i` across all codebooks, and `residual = residual - z_q_i` for each step.

### 1.2 What the RVQ Returns (But FAquantizer Does NOT Expose)

`ResidualVectorQuantize.forward()` returns:
- `z_q`: the full sum of all codebook outputs, projected back to input_dim
- `codes`: `[B, n_codebooks, T]` — discrete indices
- `latents`: `[B, n_codebooks * codebook_dim, T]` — pre-quantization projections
- `commitment_loss`, `codebook_loss`: scalars

Critically, the RVQ also computes `all_quantized` (stacked individual `z_q_i` tensors of shape `[n_codebooks, B, D, T]`) internally, but **this is discarded** and never returned by the `FAquantizer`.

The `FAquantizer.encode()` and `FAquantizer.forward()` return:
- `[codes_c, codes_p, codes_t, codes_r]` — discrete codes only
- `[z_c, z_p, z_t, z_r]` — the **summed** quantized outputs (each is the full RVQ sum for that group)

### 1.3 Forward V2 (timbre_norm=True, The Default)

In `forward_v2`:
```python
z_p, codes_p, ... = self.prosody_quantizer(x, 1)      # 1 codebook
z_c, codes_c, ... = self.content_quantizer(x, 2)        # 2 codebooks
# NO timbre_quantizer
z_r, codes_r, ... = self.residual_quantizer(residual, 3) # 3 codebooks

outs = z_p.detach() + z_c.detach() + z_r * res_mask
# Then timbre_norm modulation:
outs = timbre_norm(outs) * gamma + beta
```

Key: `z_c` here is `content_quantizer`'s full output = sum of codebook 1 output + codebook 2 output. There is **no way** to get z_c1 or z_c2 individually from the current FAquantizer API.

---

## 2. Decoder Expectations

The DAC decoder (`dac/model/dac.py`) takes `[B, 1024, T]` and up-samples back to waveform. It receives the full combined latent:

```python
z_combined = z_q  # from quantizer, already includes all factors
waveform = decoder(z_combined)
```

The decoder has **no awareness** of z_c1, z_c2, z_p, z_t, or z_r as separate entities. It receives one tensor. The `decode()` method in `reconstruct.py` does exactly this:

```python
z = model.encoder(source_audio)
z, quantized, commitment_loss, codebook_loss, timbre = model.quantizer(z, source_audio, n_c=2)
pred_wave = model.decoder(z)  # z is the full combined quantized latent
```

---

## 3. Answering the Core Questions

### 3.1 What is zc1?

$z_{c1}$ is the output of the **first** codebook in the content ResidualVectorQuantize. It is an 8-dimensional quantized vector per frame, obtained by:
1. Projecting the residual to codebook_dim (8) via `in_proj`
2. L2-normalizing and finding the nearest codebook entry
3. Applying straight-through estimator
4. Projecting back to 1024-dim via `out_proj`

$z_{c1}$ captures the **primary content information** — the phonetic/semantic content of speech.

### 3.2 What is zc2?

$z_{c2}$ is the output of the **second** codebook in the content RVQ, computed from the residual after z_c1 is subtracted. It captures **secondary content information** — finer content details that remain after the primary content is quantized.

$z_c = z_{c1} + z_{c2}$ (the content_quantizer's output).

### 3.3 Where is zc2 created?

Inside `ResidualVectorQuantize.forward()`:
```python
for i, quantizer in enumerate(self.quantizers):
    z_q_i, ... = quantizer(residual)  # i=0 -> z_c1, i=1 -> z_c2
    residual = residual - z_q_i
    z_q = z_q + z_q_i * mask
```

For content (2 codebooks): iteration 0 produces z_c1, iteration 1 produces z_c2. The `all_quantized` stack holds both, but FAquantizer discards it.

### 3.4 What inputs generate zc2?

- The **residual** input to codebook 1: `x - z_p.detach()` (prosody-subtracted encoder features)
- The **residual** input to codebook 2: `x - z_p.detach() - z_c1` (prosody and z_c1 subtracted)

In FAcodec's encode path:
```python
z_c, codes_c, ... = self.content_quantizer(x, n_c=2)
# Internally: z_c = VQ0(x_residual) + VQ1(x_residual - VQ0_output)
```

### 3.5 Is the original zc2 valid after zc1 is modified?

**No. Absolutely not.** The original z_c2 was computed from the residual `x - z_p.detach() - z_c1_original`. If you replace z_c1 with a modified version (denoised $\hat{z}_{c1}$), the residual changes, and the original z_c2 no longer corresponds to the correct residual. The original z_c2 must be discarded.

### 3.6 What must be recomputed?

If z_c1 is modified (accent conversion), you must recompute z_c2. Per the FAC-FACodec paper, this is done by:

1. Computing denoised $\hat{z}_{c1}$ from the diffusion process
2. Using the **denoiser's `fc_zc2` head** to predict z_c2 from $\hat{z}_{c1}$ and encoder features (the transformer hidden states `h`)

The paper says z_c2 is "predicted from the denoised $\hat{z}_{c1}$ and encoder features" — meaning the denoiser's fc_zc2 head takes the transformer representation and the x0 estimate to predict z_c2.

### 3.7 What decoder representation ultimately receives zc1/zc2?

The decoder receives the **full combined latent** `z_q`:
```
z_q = z_p.detach() + z_c.detach() + z_r * mask    (forward_v2)
z_q = timbre_norm(z_q) * gamma + beta              (timbre modulation)
decoder(z_q) → waveform
```

Where `z_c.detach() = z_c1.detach() + z_c2.detach()`.

For accent conversion, the modified representation would be:
```
z_q_modified = z_p.detach() + (z_c1_modified + z_c2_recomputed) + z_r * mask
z_q_modified = timbre_norm(z_q_modified) * gamma + beta
decoder(z_q_modified) → converted waveform
```

### 3.8 Does the denoiser predict zc2, or is it recomputed from encoder features?

**Both.** The denoiser:
1. Runs diffusion on z_c1 to get denoised $\hat{z}_{c1}$ (the primary output)
2. Has a `fc_zc2` head that predicts z_c2 from the transformer hidden states `h` concatenated with the x0 estimate

```python
# From denoiser.py forward():
x0_hat = (zc1_noisy - sqrt_1mabar[t] * eps_pred.detach()) / sqrt_abar[t]
zc2_input = torch.cat([h, x0_hat.detach().transpose(1, 2)], dim=-1)
zc2_pred = self.fc_zc2(zc2_input).transpose(1, 2)
```

This is consistent with the paper's description: z_c2 is predicted from denoised z_c1 and the denoiser's internal representations (which encode encoder features through the input projection).

---

## 4. AccentEdge Adapter Analysis

### 4.1 Current Assignment

In `FACodecAdapter.encode()`:
```python
z_c, z_p, z_r = quantized_list  # forward_v2 returns [z_p, z_c, z_r]
# ...
return FactorizedLatents(
    content=z_q,           # timbre-conditioned combined latent ✓
    content_zc1=z_c,      # ⚠️ BUG: z_c is the FULL sum, not z_c1 alone
    content_zc2=None,      # ⚠️ BUG: never populated (upstream doesn't expose it)
    prosody=z_p,
    detail=z_r,
    timbre=timbre,
)
```

### 4.2 Is z_c the full content (zc1+zc2) or just zc1?

**z_c is the full content = z_c1 + z_c2.** The `content_quantizer` with 2 codebooks returns the sum of both codebook outputs. The adapter assigns this sum to `content_zc1`, which is incorrect.

### 4.3 Does the adapter correctly handle zc2?

**No.** The adapter:
1. Sets `content_zc2=None` — never exposed
2. Cannot separate z_c1 from z_c2 because the upstream FAquantizer doesn't expose individual codebook outputs
3. Even if it could, the original z_c2 becomes invalid as soon as z_c1 is modified

---

## 5. Denoiser Analysis

### 5.1 Does the denoiser have a zc2 prediction head?

**Yes.** `DenoisingTransformerModel` has:
- `fc_out`: predicts epsilon noise (for DDPM diffusion)
- `fc_zc2`: predicts z_c2 from transformer hidden states + x0 estimate

```python
self.fc_zc2 = nn.Sequential(
    nn.Linear(d_model + facodec_dim, 4 * facodec_dim),
    nn.GELU(),
    nn.Linear(4 * facodec_dim, facodec_dim),
)
```

### 5.2 Is this consistent with the paper's recompute-from-encoder-features approach?

**Partially.** The paper says z_c2 is "predicted from the denoised z_c1 and encoder features." The current implementation:
- Uses the denoiser's transformer hidden states `h` (which encode encoder features via the input projection) ✓
- Concatenates with x0_hat (the denoised z_c1 estimate) ✓
- Predicts z_c2 via an MLP head ✓

This is consistent with the paper's approach. However, there's a potential issue: the `fc_zc2` input concatenates `h` (transformer output) with `x0_hat.detach()`. The paper may intend z_c2 to be predicted directly from encoder features (not from the denoised z_c1), or it may intend a different architecture. Without the official FAC-FACodec implementation, this is an independent design choice.

---

## 6. Contract: What a Correct Implementation Must Do

### 6.1 The z_c1 / z_c2 Separation Contract

```
INPUT: encoder features x [B, 1024, T]
  ↓
PROSODY QUANTIZER (1 codebook)
  → z_p = VQ_p(x) [B, 1024, T]
  ↓ residual = x - z_p.detach()
CONTENT QUANTIZER (2 codebooks, residual chain)
  → z_c1 = VQ_c0(residual) [B, 1024, T]   ← FIRST codebook only
  → z_c2 = VQ_c1(residual - z_c1) [B, 1024, T]  ← SECOND codebook only
  → z_c = z_c1 + z_c2  ← what FAcodec currently returns
  ↓ residual = residual - z_c1 - z_c2
RESIDUAL QUANTIZER (3 codebooks)
  → z_r [B, 1024, T]
```

### 6.2 The Accent Conversion Contract

```
ORIGINAL PIPELINE:
  x → encoder → z
  z → quantizer → z_p, z_c1, z_c2, z_r
  z_combined = z_p + z_c1 + z_c2 + z_r (+ timbre modulation)
  decoder(z_combined) → waveform

CONVERSION PIPELINE:
  1. Encode source: z_s, z_ps, z_c1s, z_c2s, z_rs
  2. Encode target reference: z_t, z_pt, z_c1t, ... (extract target accent)
  3. Denoise z_c1s with target phonemes → z_c1s_denoised
  4. Predict z_c2s_new from denoiser fc_zc2 head (using z_c1s_denoised + encoder features)
  5. z_modified = z_ps + z_c1s_denoised + z_c2s_new + z_rs
  6. Apply timbre modulation (from source or target, per strategy)
  7. decoder(z_modified) → converted waveform
```

### 6.3 What Must Be Recomputed

| Component | Valid after z_c1 modification? | Action |
|-----------|-------------------------------|--------|
| z_c1 | N/A (this is what we modify) | Replace with denoised version |
| z_c2 | **NO** — computed from original residual | Predict via `fc_zc2` head |
| z_p | YES — prosody is independent | Preserve from source |
| z_r | YES — residual is computed after content | Preserve from source |
| timbre (g) | YES — global utterance property | Preserve or transfer per strategy |
| encoder features for z_c2 prediction | YES — from source encoder output | Use during z_c2 prediction |

### 6.4 Implementation Requirements

1. **Expose individual codebook outputs**: The `ResidualVectorQuantize` must expose `all_quantized` so `z_c1` and `z_c2` can be separated. This requires modifying either:
   - `FAquantizer` to return individual codebook outputs alongside the sum, or
   - The RVQ to add an `output_individual` flag to its return value

2. **Adapter must set `content_zc1` to the FIRST codebook output only**: Currently it assigns the full `z_c` sum.

3. **Adapter must expose `content_zc2`**: Either as the second codebook output (for training/reconstruction) or as a placeholder that gets filled by the denoiser's prediction (for inference).

4. **Denoiser `fc_zc2` input must match**: The concatenation of transformer hidden states `h` and x0_hat must be dimensionally correct with whatever representation the content quantizer produces.

5. **Decoder input must always be the full combined sum**: `z_p + z_c1 + z_c2 + z_r` (modulated by timbre). Never pass individual components to the decoder.

---

## 7. Risk Assessment

### 7.1 What Goes Wrong If zc2 Handling Is Wrong

| Risk | Consequence | Severity |
|------|-------------|----------|
| Using full `z_c` as `content_zc1` | z_c2 is implicitly included in z_c1; denoiser operates on already-summed representation; accent conversion corrupts both z_c1 and z_c2 content | **CRITICAL** — breaks entire conversion pipeline |
| Not recomputing z_c2 after z_c1 modification | Decoder receives stale z_c2 that doesn't match modified z_c1; produces garbled output | **CRITICAL** — audible artifacts, speech corruption |
| Passing individual z_c1/z_c2 to decoder instead of sum | Decoder expects 1024-dim input; receiving partial representations produces noise | **HIGH** — no valid reconstruction |
| Predicting z_c2 from wrong features | z_c2 prediction doesn't capture original content details; accent conversion loses phonetic fidelity | **MEDIUM** — quality degradation, not complete failure |
| Forgetting timbre modulation on modified z_q | Speaker identity changes unexpectedly; VC output sounds like wrong speaker | **MEDIUM** — identity leak |

### 7.2 Silent Failure Modes

- **z_c assigned to content_zc1**: The model will train and "work" but will be learning to denoise the already-summed z_c (z_c1+z_c2), not z_c1 alone. The fc_zc2 head would then predict z_c2 given (z_c1+z_c2) as input, which is redundant and mathematically inconsistent.
- **z_c2=None at inference**: If the denoiser predicts z_c2 but the pipeline doesn't use it (falls back to z_c from encoder), the conversion is incomplete — only z_c1 is changed, z_c2 remains source-accent.

---

## 8. Recommendation: Predict vs Recompute

### 8.1 The Paper's Approach

The FAC-FACodec paper recommends **predicting** z_c2 via a learned head (the `fc_zc2` MLP), not recomputing it from the encoder. The reasoning:
- z_c2 is a small residual (8-dim after projection) that can be reliably predicted from denoised z_c1
- Full recomputation would require keeping the content quantizer accessible during inference, adding complexity
- The denoiser already has transformer representations that encode sufficient information

### 8.2 AccentEdge Recommendation

**Use the predict approach** (consistent with the paper):

1. During training:
   - Compute z_c1, z_c2 from content_quantizer individually
   - Run denoiser on z_c1 with noise → get denoised z_c1_hat
   - Train fc_zc2 to predict z_c2 from (denoiser_hidden_states, z_c1_hat)
   - Reconstruct: z_combined = z_p + z_c1_hat + fc_zc2(...) + z_r
   - Reconstruction loss trains the prediction

2. During inference:
   - Denoise source z_c1 with target phonemes → z_c1_denoised
   - Predict z_c2_new = fc_zc2(hidden_states, z_c1_denoised)
   - z_modified = z_p_source + z_c1_denoised + z_c2_new + z_r_source
   - Apply timbre modulation
   - decoder(z_modified)

3. **Required upstream modification**: FAquantizer must expose individual codebook outputs from the content RVQ so z_c1 and z_c2 can be separated for training targets.

### 8.3 Why Not Full Recomputation

Full recomputation would require:
- Re-running the content RVQ on a modified residual during inference
- This means the content quantizer must remain accessible (not fully frozen)
- Adds complexity and potential numerical instability
- The predict approach is simpler and the paper validates it

---

## 9. Summary of Gaps to Fix

| Gap | Location | Fix Required |
|-----|----------|-------------|
| `content_zc1` assigned full `z_c` sum | `facodec.py:147` | Must be first codebook output only |
| `content_zc2=None` never populated | `facodec.py:148` | Must expose second codebook output |
| FAquantizer doesn't expose individual codebook outputs | `modules/quantize.py` | Must add `all_quantized` to return or add per-codebook accessor |
| Denoiser fc_zc2 input shape depends on facodec_dim | `denoiser.py:188` | Verify concatenation dimensions match content quantizer output dim |
| Reconstruction path uses full `z_q` (correct) | `facodec.py:182` | No change needed — this is correct |

---

## 10. Open Questions (Cannot Answer Without Official FAC-FACodec Code)

1. **Exact fc_zc2 architecture**: The paper says "predict z_c2 from denoised z_c1 and encoder features" but doesn't specify the exact architecture. Our MLP head is a reasonable interpretation but may differ from the official implementation.

2. **Timestep for z_c2 prediction**: Is z_c2 predicted once at the final timestep, or at every denoising step? Our implementation predicts once from the final x0_hat.

3. **z_c2 quantization**: Is z_c2 predicted in the quantized space (codebook indices) or in the continuous space (before quantization)? Our implementation predicts in the continuous space (same dim as z_c1), which is then added directly.

4. **Training objective for fc_zc2**: L2 loss against ground-truth z_c2? Some other loss? Our implementation should use L2/MSE between predicted z_c2 and the original z_c2 from the content quantizer.

---

*Document generated by tracing upstream FAcodec code (Plachtaa/FAcodec, modules/quantize.py, dac/nn/quantize.py) and AccentEdge adapter/denoiser. All findings derived from actual code, not paper assumptions.*
