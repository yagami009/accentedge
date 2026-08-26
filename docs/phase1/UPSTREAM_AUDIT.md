# Upstream Audit — FACodec / FAC-FACodec

## FACodec (NaturalSpeech 3 codec)

| Property | Value | Status |
|---|---|---|
| Repository | https://github.com/Plachtaa/FAcodec | UPSTREAM_VERIFIED |
| License (code) | NOT SPECIFIED | LICENSE_UNKNOWN |
| Pretrained checkpoint | Plachta/FAcodec (Hugging Face Hub) | UPSTREAM_VERIFIED |
| Sample rate | 24000 Hz | UPSTREAM_VERIFIED |
| Codec input | Raw waveform | UPSTREAM_VERIFIED |
| Quantization | DAC-style residual vector quantization | UPSTREAM_VERIFIED |
| Content codebooks | 2 codebooks, size 1024, dim 8 | UPSTREAM_VERIFIED |
| Prosody codebooks | 1 codebook, size 1024, dim 8 | UPSTREAM_VERIFIED |
| Timbre codebooks | 2 codebooks, size 1024, dim 8 | UPSTREAM_VERIFIED |
| Residual codebooks | 3 codebooks, size 1024, dim 8 | UPSTREAM_VERIFIED |
| Content extract API | model.encoder(waveform) → z, then model.quantizer(z, waveform, n_c=2) → codes | UPSTREAM_VERIFIED |
| Decode API | model.decoder(z) → waveform | UPSTREAM_VERIFIED |
| Reconstruction tested | Yes, 10 L2-ARCTIC clips | DIAGNOSTIC_ONLY |
| Reconstruction quality | Duration preserved (ratio 0.99), but ECAPA identity sim 0.05-0.24 | INVALIDATED_BY_ADAPTER_BUG — results corroborate broken speaker pathway (0.05–0.24 expected from severe speaker-path failure). Results will be re-evaluated after adapter fix. A large jump from 0.05–0.24 range would confirm the diagnosis. HEURISTIC EXPECTATION (not threshold): post-fix reconstruction should enter high same-speaker region. Acceptance will be calibrated against same-speaker/impostor ECAPA distributions. |

### Verified API

```python
from modules.commons import build_model, recursive_munch
from hf_utils import load_custom_model_from_hf
import yaml

ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")
config = yaml.safe_load(open(config_path))
model_params = recursive_munch(config['model_params'])
model = build_model(model_params)
ckpt_params = torch.load(ckpt_path, map_location="cpu")
for key in ckpt_params:
    model[key].load_state_dict(ckpt_params[key])
```

Encode:
```python
z = model.encoder(waveform[None, ...].float())
codes, quantized, _, _, timbre = model.quantizer(z, waveform[None, ...].float(), n_c=2, return_codes=True)
# codes[0] = content codes (2 codebooks)
# codes[1] = prosody codes (1 codebook)
# quantized[0] = z_c (content quantized)
# quantized[1] = z_p (prosody quantized)
# quantized[2] = z_t (timbre quantized)
# quantized[3] = z_r (residual quantized)
# timbre = global timbre embedding
```

Decode:
```python
z_combined = z_p + z_c + z_t + z_r  # quantized sum
reconstructed = model.decoder(z_combined)
```

## FAC-FACodec (accent conversion method)

| Property | Value | Status |
|---|---|---|
| Official implementation | NOT FOUND | NO-SUCH-CODE |
| Paper | arxiv:2510.10785v2 | PAPER_EXPLICIT |
| Checkpoints | NOT FOUND | NO-SUCH-CODE |
| Training data | LJSpeech (24h, single female speaker) | PAPER_EXPLICIT |
| Code license | arXiv perpetual non-exclusive | UPSTREAM_VERIFIED |
| Weights license | NOT SPECIFIED | LICENSE_UNKNOWN |

**FAC-FACodec official implementation: NOT AVAILABLE / NOT FOUND.**

Our implementation will be called: **AccentEdge FAC-FACodec Reimplementation**.

We are reimplementing from the paper description using the FACodec encoder/decoder from Plachtaa/FAcodec.
