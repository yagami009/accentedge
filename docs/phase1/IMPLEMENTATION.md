# AccentEdge Phase 1 — Implementation Plan

## Status: Phase 1 Core Complete (CPU Tests Passing)

### What we've done (local)
1. **Evidence audit** — verified FAC-FACodec paper details, upstream FAcodec API, transcript dependency
2. **Colab bootstrap** — `scripts/colab_bootstrap.sh`, `scripts/verify_cuda.py`
3. **Codec interface** — `codec/interfaces.py` (FactorizedLatents, FactorizedSpeechCodec)
4. **FACodec adapter** — `codec/facodec.py` (wraps Plachta/FAcodec)
5. **Denoiser** — `phase1/denoiser.py` (paper-faithful, 11 tests passing)
6. **Diffusion math** — `phase1/diffusion.py` (schedule, q_sample, DDIM, strength)
7. **Strength control** — `phase1/strength.py`
8. **Evaluation** — `evaluation/{content,identity,acoustic}.py`
9. **Configuration** — `pyproject.toml`

### What's next (Colab CUDA)
1. Run `scripts/verify_cuda.py` on Colab
2. Run `scripts/reconstruct.py` to verify FACodec reconstruction
3. Run `scripts/train_phase1.py --config configs/phase1/smoke.yaml` (smoke train)
4. Run `scripts/train_phase1.py --config configs/phase1/overfit.yaml` (tiny overfit)
5. Run `scripts/sweep_strength.py` (strength sweep 0/0.25/0.5/0.75/1.0)
6. Listen and decide Phase 2
