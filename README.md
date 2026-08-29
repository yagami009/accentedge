# AccentEdge

**Identity-preserving speech-to-speech accent conversion for Indian English → US-neutral.**

Research project investigating whether direct, offline accent transformation is technically feasible for BPO call-centre environments.

## What This Is

A monorepo containing all phases of the AccentEdge research program:

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 0** — Target Feasibility | In progress | TGFP v2 protocol: can we construct a usable accent transformation target? |
| **Phase 1** — BPO Benchmark | Scaffolded | Speaker-disjoint, leakage-resistant evaluation instrument |
| **Phase 2** — Architecture Bake-off | **Complete** | Candidate D (Minimal Hybrid) selected as streaming architecture |
| **Phase 3** — AccentEdge S2S Model | Planned | First proprietary model work |
| **Phase 4** — Streaming Inference | Not started | Real-time streaming conversion |
| **Phase 5** — Optimisation | Not started | Latency, memory, quality tuning |
| **Phase 6** — Runtime | Not started | Windows endpoint, admin portal |
| **Phase 7** — Pilot | Not started | BPO deployment pilot |
| **Phase 8** — Production | Not started | Full production rollout |

## Phase Roadmap

```
Phase 0: Target Feasibility (Step 0)
    ↓ passes
Phase 1: BPO Benchmark
    ↓
Phase 2: Architecture Bake-off  ← DONE (Candidate D selected)
    ↓
Phase 3: AccentEdge S2S Model
    ↓
Phase 4: Streaming Inference
    ↓
Phase 5: Optimisation
    ↓
Phase 6: Runtime / Windows endpoint
    ↓
Phase 7: Pilot / BPO deployment
    ↓
Phase 8: Production
```

## Monorepo Structure

```
accentedge/
├── src/accentedge/
│   ├── phase0/          ← TGFP v2 experiment framework
│   ├── phase1/          ← FAC-FACodec model (diffusion + denoiser)
│   ├── benchmark/       ← BPO benchmark evaluation suite
│   ├── codec/           ← FACodec adapter (factorized latents)
│   ├── evaluation/      ← STOI, PESQ, MCD, speaker sim, WER
│   ├── models/          ← 5 streaming candidates (A/B/C/D + Sparse Repair)
│   ├── streaming/       ← Virtual-time simulator, chunker, latency
│   ├── training/        ← Dataset, checkpoint, overfit, trainer
│   ├── config/          ← schema + loader
│   ├── data/            ← lineage, schema, validation
│   ├── experiments/     ← registry
│   ├── profiling/       ← latency, memory
│   ├── reporting/       ← HTML/JSON reports, ADR generation
│   ├── audio/           ← buffer, capture, playback, VAD primitives
│   └── cli/             ← CLI entry point
├── FAcodec/             ← Plachta/FAcodec (symlink)
├── Amphion/             ← Amphion TTS toolkit (for Colab)
├── checkpoints/         ← Model weights
├── data/                ← Audio samples, gold targets, manifests
├── configs/             ← YAML configs per phase
├── docs/                ← Specs, audits, decision records
├── scripts/             ← Entry points for each phase
├── tests/               ← Unified test suite
└── pyproject.toml       ← Single dependency lock
```

## What This Is NOT

Not a deployable system. Not a streaming inference engine. Not a Windows runtime or admin portal. This is research infrastructure for validating an accent conversion thesis.

## Setup

```bash
pip install -e .
```

For GPU training:
```bash
pip install -e ".[cuda]"
```

For MPS (Apple Silicon):
```bash
pip install -e ".[mps]"
```

## Key Documents

- `docs/phase0/TGFP_V2.md` — Target Generation Feasibility Protocol
- `docs/phase0/PHASE_0_SPEC.md` — Gate sequence and decision tree
- `docs/phase1/ARCHITECTURE_DECISIONS.md` — Phase 2 bake-off result (Candidate D selected)

## License

MIT
