# Paper Verification — FAC-FACodec (arxiv:2510.10785v2)

Source: https://arxiv.org/html/2510.10785

## Verified from Paper

### 3.1 Factorization

| Component | Paper Name | Verified | Notes |
|---|---|---|---|
| Content | $z_{c1}$ (quantized content residual) | PAPER_EXPLICIT | 8-dimensional, quantized |
| Content residual | $z_{c2}$ | PAPER_EXPLICIT | Predicted from denoised $\hat{z}_{c1}$ + encoder features |
| Prosody | $z_p$ | PAPER_EXPLICIT | Passed through during conversion |
| Acoustic detail | $z_d$ | PAPER_EXPLICIT | Passed through during conversion |
| Global timbre | $g$ | PAPER_EXPLICIT | Passed through during conversion |

**Important**: The paper's actual factorization is $z_c = z_{c1} + z_{c2}$, not the four independent factors the prompt assumes. $z_{c2}$ is NOT preserved during conversion — it is recomputed from denoised $\hat{z}_{c1}$ and encoder features.

Frame rate: 20 ms per frame (50 fps).
Content dimensionality: 8-dim quantized vectors.
Codebook size: 1024, codebook dim: 8 (verified from upstream FACodec config).

### 3.2 Diffusion Details

| Detail | Value | Status |
|---|---|---|
| Target representation | $z_{c1}$ (content residual) | PAPER_EXPLICIT |
| Predicts | noise $\varepsilon$ | PAPER_EXPLICIT |
| Loss | $\|\varepsilon - s_\theta(x_t, t, \pi)\|_2^2$ | PAPER_EXPLICIT |
| Noise schedule | Linear $\beta_t \in [10^{-4}, 2\times10^{-2}]$ | PAPER_EXPLICIT |
| Timesteps | $T = 100$ | PAPER_EXPLICIT |
| Denoiser architecture | 6-layer Transformer, 8 heads, dim 1024, FFN 2048, dropout 0.1 | PAPER_EXPLICIT |
| Conditioning | Phoneme embeddings $\pi$ via FiLM + additive embeddings | PAPER_EXPLICIT |
| Inference algorithm | DDIM ODE formulation, $K=100$ steps | PAPER_EXPLICIT |
| $t_{start}$ meaning | Initial timestep for partial denoising | PAPER_EXPLICIT |

### 3.3 Transcript Dependency

**Transcripts/phonemes ARE required during inference.**

- Phonemes extracted via phonemizer + eSpeak-ng
- Aligned with Wav2Vec2 XLSR
- Fed to denoiser as conditioning $\pi$ via FiLM
- No speech-only inference path described

### 3.4 Released Assets

| Asset | Status |
|---|---|
| FAC-FACodec implementation | NOT AVAILABLE / NOT FOUND |
| FACodec implementation | AVAILABLE (Plachtaa/FAcodec, no license file) |
| FACodec checkpoint | AVAILABLE (Plachta/FAcodec on HF Hub, ~869MB) |
| FAC-FACodec checkpoints | NOT AVAILABLE |
| FAC-FACodec paper | AVAILABLE (arxiv:2510.10785v2) |

**Our implementation is an independent reimplementation, not an official release.**

### 3.5 Key Deviation from Prompt Assumptions

The prompt assumes four independent factors (content, prosody, acoustic detail, timbre) that can all be "preserved." The paper actually has content split into TWO residuals ($z_{c1}$ + $z_{c2}$), where $z_{c2}$ is NOT preserved but recomputed. This means:

1. The "preserve all other factors" invariant needs to preserve: $z_p$, $z_d$, $g$, AND encoder features for $z_{c2}$ recomputation.
2. The internal codec interface should reflect $z_c = z_{c1} + z_{c2}$ accurately.
