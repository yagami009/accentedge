# AccentEdge Phase 1 — Live Status

_Last updated: 2026-08-26_

## Milestone Progress

| Milestone | Status | Details |
|-----------|--------|---------|
| **FACodec reconstruction** | 🔄 READY | `scripts/verify_facodec_direct.py` ready for Colab T4 |
| **Module structure** | ✅ Done | phase1/ (diffusion, denoiser, strength), evaluation/ (acoustic, content, identity, phonemes) |
| **Adapter (FACodecAdapter)** | ⚠️ Needs fix | Using Amphion FACodec; needs decoder.inference API |
| **Reconstruction gate** | 🔄 READY | Script tests encode→decode round-trip with mel L1 metric |

## What Just Changed

- **`verify_facodec_direct.py`** — New verification script
  - Uses upstream FAcodec (same path as FAC-FACodec training)
  - Mock audiotools (unavailable on Colab)
  - Downloads LibriSpeech test-clean, encodes → decodes, computes mel L1
  - Gate: all mel L1 < 0.15

- **Repo cleanup** — Removed 1.5GB checkpoint blobs from git history (repo now 75KB)

## Next Steps

1. Run `colab run scripts/verify_facodec_direct.py --gpu T4` to verify reconstruction
2. Fix FACodecAdapter decode to use `decoder.inference()` API
3. Run tiny native-latent overfit
4. Run Indian-English inference through Phase 1 pipeline
