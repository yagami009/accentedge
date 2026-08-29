# AccentEdge Phase 0 — Target & Contract Feasibility

Phase 0 is not the first version of AccentEdge. It is not the streaming model, not the real-time engine, not the BPO client, not the Windows runtime, and not even the dataset-building phase.

Phase 0 exists to answer one question:

> Can we define and produce an offline speech target that represents what AccentEdge should eventually generate, while preserving the speaker and meaning well enough that we would deliberately train a weaker causal real-time model to imitate it?

Everything else is downstream.

---

## What Phase 0 is really proving

Assume an Indian-English BPO agent says:

> "I can see a charge of thirty dollars posted on the thirteenth of August."

Before we think about 150 ms latency, causal encoders, DDSP, Windows endpoints or BPO deployment, we need to know what the ideal transformed waveform should sound like.

We want something approximately like:

```
SOURCE
same person, Indian-English realization, correct words,
original emotion, original conversational intent
        ↓
IDEAL ACCENTEDGE TARGET
same person, same words, same emotional intent,
meaningfully more US-neutral pronunciation,
natural speech, timing still compatible with conversation
```

Phase 0 asks whether that second waveform is actually constructible. And it asks something even deeper:

> Is our definition of the desired transformation linguistically coherent?

TGFP v2 explicitly acknowledges that the current v1 contract itself is under test because segmental pronunciation, rhythm, intonation and timing may not be as separable as we would like.

So Phase 0 simultaneously tests: the target, the metrics, and the product definition.

---

## What Phase 0 does NOT attempt

| Question | Phase 0? |
|---|---|
| Can it run in real time? | No |
| Can it run below 200 ms? | No |
| Can it run on CPU? | No |
| Which streaming encoder should we use? | No |
| Should we use DDSP or HiFi-GAN? | No |
| What will the Windows client look like? | No |
| Does WebRTC work? | No |
| What BPO dashboard do we need? | No |
| How do we price it? | No |
| Can we create a valid target? | YES |
| Does it still sound like the same person? | YES |
| Did we preserve words exactly? | YES |
| What pronunciation dimensions should change? | YES |
| What should remain unchanged? | YES |
| How much timing movement is natural? | YES |
| How much speaker-embedding movement is natural? | YES |
| Can synthetic targets approximate human cross-accent speech? | YES |
| Full conversion or sparse repair? | YES |

---

## Phase 0 output

Phase 0 does NOT end with a model. It ends with a decision.

| Outcome | Meaning |
|---|---|
| FULL-S2S PASS | Good whole-speech targets can be created; proceed toward causal direct S2S |
| SPARSE-REPAIR PASS | Full transformation damages identity, but controlled pronunciation repair works |
| TEACHER FAIL / GOLD PASS | Humans demonstrate the desired transformation, but our synthetic supervision cannot reproduce it yet |
| FUNDAMENTAL FAIL | Even natural same-speaker cross-accent behavior cannot satisfy the intended product contract |

---

## Final Phase 0 gate sequence

```
PHASE 0
├── Gate -1A    Source annotation validity
├── Gate -1B    Measurement/probe validity
├── Gate 0      One-shot target sanity check
├── Gate 1A     Generate candidate teacher targets
├── Gate 2      Natural cross-accent calibration
├── Contract checkpoint    Is our v1 linguistic contract human-realistic?
├── Gate 1B     Adjudicate generated targets using gold-derived criteria
└── Phase-0 Decision
    ├── FULL-S2S PASS
    ├── SPARSE-REPAIR PASS
    ├── TEACHER FAIL / GOLD PASS
    └── FUNDAMENTAL FAIL
```

---

## Gate -1A — Source validity

> Do we actually know what was pronounced, where it was pronounced, and whether each targeted token needed correction in the first place?

Procedure:
1. Raw speech → exact human transcript
2. Initial forced alignment
3. Manual boundary correction
4. Canonical target phones
5. Observed source realization
6. Target-feature labels
7. ALREADY-TARGET / DEVIANT / AMBIGUOUS

---

## Gate -1B — Measurement validity

Before using automated accent metrics, verify those metrics can actually distinguish what they claim to measure.

Probe-validity test:
- Known natural US token → probe → should look target-like
- Known deviant Indian realization → probe → should look substitute-like
- Already-target Indian realization → probe → should look target-like

---

## Gate 0 — One-shot target sanity check

One speaker. One sentence. Strategy B. Built manually. Listen.

```
Indian source speech
       ↓
native-US linguistic realization
       ↓
source identity / timbre transfer
       ↓
candidate target
```

Gate 0 passes if Strategy B produces an example where a reasonable listener can plausibly interpret the result as the original human with modified pronunciation. It does not need to meet full teacher criteria. It simply must avoid catastrophic identity failure.

---

## Gate 1A — Generate candidate teacher targets

Generate A/B/C candidates. No pass/fail judgment yet — gold calibration comes first.

### Strategy A — Source-conditioned native synthesis
Transcript + source F0 + source duration + source speaker embedding → US-target-conditioned synthesis

### Strategy B — Native realization first, identity second
Transcript → native-US realization → identity/timbre transfer → source speaker target

### Strategy C — Sparse control-domain repair
Whole source waveform → latent/acoustic analysis → edit selected pronunciation parameters → whole-utterance synthesis

---

## Gate 2 — Natural cross-accent calibration

Six-condition control design:

| Condition | Speaker | Accent | Difference |
|---|---|---|---|
| A | same | Indian | baseline |
| B | same | Indian | different session |
| C | same | Indian | different rate/emotion |
| D | same | US/code-switched | actual cross-accent condition |
| E | different | Indian | identity impostor |
| F | different | US | identity + accent impostor |

---

## Contract checkpoint

After gold is collected, pause. Ask: Is our linguistic contract still coherent?

The contract passes if natural speakers demonstrate that targeted segmental/stress changes can occur while identity/emotion/timing properties remain within acceptable human ranges.

---

## Gate 1B — Target adjudication

Every A/B/C target evaluated against:
- SOURCE: Indian speech
- TARGET CANDIDATE: A / B / C
- GOLD: same person's natural US rendering

---

## Phase-0 decision

Only four outcomes are valid:
1. FULL-S2S PASS
2. SPARSE-REPAIR PASS
3. TEACHER FAIL / GOLD PASS
4. FUNDAMENTAL FAIL

No fifth outcome called "continue because we've already spent time."

---

## Target Core Set

30 annotated source utterances for A/B/C target generation, balanced across:
- realistic BPO speech
- contrast-dense speech
- spontaneous speech
- already-target-dense speech
- critical-entity speech

---

## Linguistic contract (v1, under test)

| Dimension | v1 intention |
|---|---|
| Rhoticity | Transform |
| Selected vowel quality | Transform |
| Vowel reduction | Transform if feasible |
| /θ ð/ | Transform |
| Stop aspiration | Transform |
| Retroflex/alveolar differences | Transform |
| /v/–/w/ | Transform where needed |
| Intervocalic /t/ flapping | Transform |
| Cluster epenthesis | Transform |
| Lexical stress | Bounded correction |
| Phone duration | Allowed to change |
| Word duration | Bounded empirically |
| Phrase timing | Prefer preservation |
| Global rhythm | Out of v1 unless gold forces reconsideration |
| Global intonation | Out of v1 unless gold forces reconsideration |
| Timbre | Preserve |
| Voice quality | Preserve |
| Pitch range | Preserve broadly |
| Emotion | Preserve |
| Words | Preserve exactly |

---

## Per-token realization labels

Every in-scope token receives:
- **ALREADY-TARGET** — realization is acceptably compatible with target
- **DEVIANT** — differs meaningfully on evaluated dimension
- **AMBIGUOUS** — annotator cannot confidently decide

---

## Correction and damage rates

```
Correction rate = deviant tokens moved toward target / all deviant tokens
Damage rate = already-correct tokens made worse / all already-correct tokens
```

---

## Phase-0 time constraint

10-week side-project timebox. If evaluating the idea requires building half a research institution before producing an answer, the methodology itself has become unsuitable.

---

## References

- TGFP_V2.md — Target Generation Feasibility Protocol v2 (detailed gate specifications)
- FORENSIC_AUDIT_2026-08-25.md — Prior prototype audit (research artifact)
