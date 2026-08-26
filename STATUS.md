# AccentEdge Phase 1 — Live Status

_Last updated: 2026-08-26_

## Milestone Progress

| Milestone | Status | Details |
|-----------|--------|---------|
| **FACodec reconstruction** | 🔄 READY | Script `verify_facodec_direct.py` uses FAC-FACodec's proven init path |
| **Module structure** | ✅ Done | phase1/ (diffusion, denoiser, strength), evaluation/ (acoustic, content, identity, phonemes) |
| **Adapter (FACodecAdapter)** | ✅ Done | Uses Amphion's FACodec, timbre as FiLM conditioning |
| **Reconstruction gate** | 🔄 READY | `scripts/verify_facodec_direct.py` → Colab T4 |

## What Just Changed

- **`verify_facodec_direct.py`** — New script that tests FACodec round-trip reconstruction
  - Uses FAC-FACodec's proven `init_facodec_models()` pattern
  - Downloads LibriSpeech test-clean, encodes → decodes, computes SNR
  - Gate: all SNR > 5dB

- **`FACodecAdapter`** — Refactored to use Amphion's bundled FACodec
  - Handles the API differences between Amphion and upstream FAcodec
  - Timbre passed as FiLM conditioning to decoder

- **Repo cleanup** — Removed 1.5GB checkpoint blobs from git history

## Next Steps

1. Run `colab run scripts/verify_facodec_direct.py --gpu T4`
2. Verify SNR > 5dB on all test samples
3. If SNR passes: mark reconstruction gate as ✅
4. If SNR fails: investigate adapter reconstruction formula
