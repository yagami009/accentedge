# AccentEdge Master Dossier — Deep Analysis

**Source:** AccentEdge_Master_Dossier_2026-08-27.docx
**Sections analyzed:** 27
**Total word count:** 7319

---


## Preamble
*(5 paragraphs, 43 words)*

AccentEdge
MASTER PROJECT DOSSIER
Technology • Model Research • Data • Runtime • Deployment • Commercialization • Evidence History
Prepared as a complete handoff/reference for future engineering, research, commercial discussions, investor/customer conversations, and coding-agent sessions.
Version 1.0  |  As-of: 27 August 2026 (Asia/Kolkata)


## 0. Document Control, Status Labels & Evidence Hierarchy
*(50 paragraphs, 431 words)*

This dossier intentionally does not flatten every prior statement into one “final architecture.” AccentEdge has gone through research resets and several architectural explorations. The master therefore uses status labels and an evidence hierarchy.

## 0.1 Evidence hierarchy
1. PROJECT_CONTEXT.md / current master dossier
2. Current phase decision document (TGFP v2 for Phase 0)
3. Current experimental evidence / artifacts
4. Current code
5. Older audits, reports, milestone labels
6. Old agent summaries and exploratory architecture discussions
The authoritative internal project context explicitly states that old reports and code are not automatically more authoritative than current product/spec decisions. The immediate scientific question is still whether one convincing same-speaker target can be produced.

## 0.2 Claims that must not be carried forward as current facts
“Phase 2 complete.”
“Architecture bake-off complete.”
“Candidate D selected as the winner.”
“Speaker preservation validated.”
“Content preservation validated.”
“Accent conversion validated.”
“Real-time validated.”
“RTF proves AccentEdge is fast.”
“Benchmark scientifically validated.”
“Stage A proves same-speaker accent conversion.”

## 0.3 What “everything” means in this master
Current product thesis and linguistic contract.
All prior relevant AccentEdge/Accent Translator engineering branches: Seed-VC, FACodec, Stage A, bake-off/reset, target-generation protocol.
Every model family discussed in this thread and the commercial/license caveats attached to it.
The complete streaming/runtime architecture proposals, cross-platform strategy, on-prem hardware, deployment topology, model-plugin approach, and framework stack.
Business-model evolution from $1/min cloud service to seat-based local/on-prem deployment, including the Sanas benchmark.
Every dataset/source discussed, its intended role, license posture, provenance caveat, and link.
Evaluation gates, RTF/latency definitions, critical-content safety, speaker identity, correction/damage metrics, and listening-study design.
Explicit unknowns, risks, decisions not yet made, and immediate next actions.
A consolidated link register at the end.

## 0.4 Master content map
1. Executive Position — What AccentEdge Is and Is Not
2. Business Thesis & Pricing Evolution
3. Product and Linguistic Contract
4. Scientific Gate: TGFP v2
5. Roadmap: Phase 0 to Controlled BPO Pilot
6. Legacy Work: What Was Built, What Failed, What Survives
7. Model Landscape: Offline, Accent-Specific, Real-Time VC
8. Existing Real-Time Model Shortcut Strategy
9. Proprietary Model Architecture (Only If Needed)
10. Streaming Audio & Runtime Architecture
11. Cross-Platform Desktop Architecture
12. On-Prem / Production Hardware & Deployment
13. Frameworks, Languages & Production Tech Stack
14. Data Strategy & Dataset Catalogue
15. Proprietary AccentEdge-PAIR Dataset
16. Evaluation, Benchmarks & Production Gates
17. Licensing, Provenance, Privacy & Enterprise Security
18. Business Economics, Competitive Proof & Go-to-Market
19. Risk Register & Unresolved Questions
20. Immediate Execution Plan & Decision Tree
Appendix A. Exact Phase-0 Test Material
Appendix B. Repository/API/Interface Blueprints
Appendix C. Chronological Decision Log
Appendix D. Complete Link Register


## 1. Executive Position — What AccentEdge Is and Is Not
*(15 paragraphs, 221 words)*


## 1.1 First commercial wedge

## 1.2 Production path vs. paths explicitly rejected
INTENDED
LIVE SPEECH IN
    ↓
causal / bounded-context speech representation
    ↓
pronunciation/accent transformation
    ↓
LIVE SPEECH OUT

NOT THE DEPLOYED CRITICAL PATH
speech → ASR → text → LLM → TTS
ASR, forced alignment, phoneme labels, TTS, voice conversion, teacher models, and text may still be used offline for training, target generation, supervision, diagnostics, and evaluation.

## 1.3 What the MVP is
Physical Headset Mic
        ↓
AccentEdge Runtime
        ↓
Real-Time S2S Model / selected engine
        ↓
AccentEdge Virtual Microphone
        ↓
Genesys / Five9 / NICE / Teams / Zoom / Webex / browser dialer / SIP softphone
        ↓
Customer
The technical MVP is not an API demo and not a research notebook. A real person must install it, choose their microphone, select “AccentEdge Microphone” in an existing calling application, talk continuously, and retain the original call if conversion fails.

## 1.4 What is not currently proven
A commercially acceptable same-speaker target exists.
An existing real-time VC model can learn same-speaker accent normalization.
The FACodec path correctly disentangles and reconstructs all required attributes end-to-end.
A streaming model meets BPO latency on unseen speakers.
CPU-only endpoint inference is acceptable on ordinary BPO PCs.
BPO agents will voluntarily keep the transformation enabled.
A BPO will pay the target price at scale.


## 2. Business Thesis & Pricing Evolution
*(22 paragraphs, 362 words)*


## 2.1 Original business idea preserved
Initial idea discussed: sell real-time accent conversion to BPOs as a service at approximately $1 per converted minute. The product would process agent audio and return transformed audio during the call.
BPO Agent → audio → AccentEdge API/GPU → converted audio → customer
Billing hypothesis: $1 / converted minute

## 2.2 Why the idea felt threatened
Research-grade models such as CosyAccent/TokAN are not necessarily true streaming engines out of the box.
Cloud inference adds network latency, privacy review, concurrency capacity, GPU cost, and outage exposure.
A BPO may not allow live call audio to leave its network.
Concurrency economics must be benchmarked by RTF and not guessed from parameter count.

## 2.3 Why real-time itself is not the deal-breaker
Sanas provides current commercial proof that real-time accent translation is possible for contact centers. Its documentation says Accent Translation processes conversation data and voice samples on-device in real time, retains personalized voice characteristics, and integrates as a desktop application. The user guide says audio is processed on the device while internet connectivity is used for access verification.
Sanas Accent Translation: https://help.sanas.ai/docs/accent-translation — Current product documentation: real-time, personalized voice, on-device processing.
Sanas App User Guide: https://help.sanas.ai/v1/docs/using-sanas-app — Desktop application; local processing with internet for access verification.
Sanas product page: https://www.sanas.ai/accent-translation — Market proof and contact-center integration positioning.

## 2.4 The bigger commercial problem with $1/minute
The public AWS Marketplace listing currently shows a 12-month Accent Translation contract for 100 users at $72,000, i.e. $60 per user per month before any negotiated enterprise terms. This is a much lower effective price than $1/minute for agents who talk for hours per day.
Sanas AWS Marketplace: https://aws.amazon.com/marketplace/pp/prodview-mr5luwansb4k6 — Public listing: 100 Accent Translation users = $72,000/12 months at time of verification.

## 2.5 Current business-model direction

## 2.6 Why on-device/local-first improves economics
Inference cost is paid largely by the endpoint hardware rather than a central GPU per minute.
Call audio can remain on the agent PC or inside the BPO LAN.
Control plane can be light: licensing, model distribution, configuration, fleet health, analytics/telemetry.
A 1,000-seat deployment at an illustrative $40/seat/month is $40,000 MRR without processing every minute in your own cloud.


## 3. Product & Linguistic Transformation Contract
*(13 paragraphs, 149 words)*


## 3.1 Why “remove the Indian accent” is the wrong specification
Accent is not an independent knob. It overlaps with phone realization, vowel quality, stress, duration, rhythm, intonation, pitch contour, coarticulation, and speaker/style cues. “Indian English” is also not one accent; speakers vary by first-language substrate, region, education, and existing target-like realizations.

## 3.2 Likely in-scope pronunciation dimensions

## 3.3 Explicitly not automatic v1 scope
Wholesale sentence-rhythm conversion.
Full US-style global intonation remapping.
Major phrase-level reconstruction.
Lexical variant substitutions such as schedule/route/advertisement/privacy/mobile as a discrete “American word choice” mechanism.
Language translation.
Speaker impersonation or voice cloning.

## 3.4 Identity definition
Identity is perceptual and physiological/timbral: listeners should continue to hear the output as the same human even when selected pronunciation patterns change. It is not defined as “maximize one arbitrary speaker-encoder cosine.” The current Phase-0 protocol calibrates identity against natural same-speaker cross-accent change.

## 3.5 Preservation vs transformation contract


## 4. Scientific Gate — Target Generation Feasibility Protocol (TGFP v2)
*(44 paragraphs, 635 words)*


## 4.1 Gate structure
The protocol timeboxes Gates −1 through 2 to approximately 10 weeks of side-project time. If the evaluation apparatus itself overruns badly, that is evidence to re-scope rather than perpetually expanding the protocol.

## 4.2 Gate −1: annotation validity
Hand-transcribe all Phase-0 utterances.
Auto-align only as initialization; manually correct boundaries in the in-scope pronunciation regions.
Track how much alignment correction was required; poor accented-speech alignment is itself a future data-cost finding.
For each in-scope token label ALREADY-TARGET, DEVIANT, or AMBIGUOUS.
Correction rate is computed only over DEVIANT tokens; damage rate only over ALREADY-TARGET tokens.
Report AMBIGUOUS count. Around >15% was discussed as a warning that annotation has become the measurement bottleneck.
Target at least 20 DEVIANT and 20 ALREADY-TARGET tokens per speaker so both correction and damage are measurable.

## 4.3 Initial source speakers and recordings
Three speakers across different Indian-English substrates: e.g. Hindi-L1 North, Dravidian-L1 South, and one contrasting background.
Moderate-to-strong accent preferred for the feasibility protocol; a mild-accent speaker can make the test falsely easy.
16 kHz or better, mono, quiet room, consistent headset/microphone.
One speaker should repeat a session on a different day to estimate session nuisance.

## 4.4 Degradation conditions

## 4.5 Gate 0: highest-information experiment
ONE SPEAKER • ONE SENTENCE • STRATEGY B • BUILT BY HAND

Indian source
   ↓
native-US linguistic realization
   ↓
transfer source identity/timbre
   ↓
candidate target

Question: “the same person speaking differently” or “a different person wearing their voice characteristics”?

## 4.6 Gate 1 teacher strategies

## 4.7 Sparse repair is not automatically “easier”
Discrete failure: one repaired word can sound abruptly different from surrounding speech.
Detection requires streaming phone recognition/deviation detection — the hard representation problem moves rather than disappears.
Lookahead can be worse because the system must decide to fire before a word is complete.
Damage rate may dominate correction rate; false repair of already-correct speech is highly salient to agents.

## 4.8 Gate 2: natural cross-accent calibration
Recruit genuine code-switchers/register shifters rather than people merely performing an American accent. A performer may change persona, voice quality, rate, loudness, and emotion, contaminating the “cost of accent change.” The protocol targets 5–10 speakers and 50–100 matched sentences once initial feasibility advances.

## 4.9 Gold-derived identity budget
d_gold = mean distance over condition D
         = same speaker: Indian ↔ naturally code-switched US

PASS target identity if:
    d(source, generated_target) ≤ d_gold + confidence interval

Use ≥3 independent speaker encoders (e.g. ECAPA-TDNN, WavLM verification head, another)
and human same-person ratings. If encoders disagree, humans decide and disagreement is recorded.

## 4.10 Accent metric: correction AND damage
Correction rate = DEVIANT tokens that became target-like / all DEVIANT tokens
Damage rate     = ALREADY-TARGET tokens that became non-target / all ALREADY-TARGET tokens
Off-target movement = changed tokens that moved toward neither desired centroid
The protocol explicitly rejects one black-box “accent score” as the headline. A conversion-strength parameter should trace a correction/damage tradeoff curve, and a conservative operating point may be preferable for BPO use.

## 4.11 Per-dimension diagnostics discussed

## 4.12 Timing is hierarchical, not one arbitrary percentage

## 4.13 Human listening study
US-native panel (~8–12) for accent/naturalness.
Identity panel (~8–12), including people unfamiliar with the source speakers.
Two independent content transcribers for critical entities.
Fully randomized/unlabelled stimuli with hidden source (low) and natural cross-accent reference (high) anchors.
Identity blocks include true-same and true-different catch trials; compute per-rater discrimination (d′) and pre-register exclusion rules.
Sessions capped around 20 minutes to reduce fatigue.
Identity paired AB 1–5 + free-text “what changed”; accent 5-point anchored; naturalness MOS; emotion match + confidence.
If confidence intervals overlap for close teacher strategies, advance both rather than overclaim a winner from a tiny sample.

## 4.14 Teacher headroom and outcomes
TGFP v2 treats earlier numerical bars as provisional unless gold-derived. The following numbers were explicitly retained as [P] engineering/admission hypotheses, not universal truths.


## 5. Delivery Roadmap — Phase 0 to Controlled BPO Pilot
*(45 paragraphs, 544 words)*

PHASE 0 — VALID TARGET
      ↓
PHASE 1 — LEARNED BASELINE
      ↓
PHASE 2 — GENERALIZATION & QUALITY
      ↓
PHASE 3 — TRUE STREAMING
      ↓
PHASE 4 — CPU / ENDPOINT RUNTIME
      ↓
PHASE 5 — PILOT-READY VIRTUAL-MIC MVP
      ↓
PHASE 6 — CONTROLLED BPO PILOT

## 5.1 Phase 0 — Valid Target
Audit provenance of every legacy source/target/prediction pair.
Replace Griffin-Lim-based listening with a compatible neural vocoder where needed.
Replace homemade speaker metric with established verification + humans.
Use WER/CER plus human critical-entity checking for content.
Produce one valid same-speaker target; listen; only then expand to ~10–30 pairs.
Phase-0 artifacts: source/, targets/, metadata.csv, target_provenance.md, listening_notes.md, PHASE0_DECISION.md.

## 5.2 Phase 1 — Learned Baseline
Do not restart a four-model architecture bake-off. Pick one simple baseline that is easy to overfit/debug.
Candidate D / Minimal Hybrid may be reused only as Baseline v0, not as a proven winner.
Overfit 5–10 valid pairs. If the model cannot memorize direction, do not scale data; fix architecture/loss/target/vocoder/training.
Then ~3–10 speakers and 100–500 paired utterances; train/evaluation speakers disjoint.
Evaluate WER/CER/entities, established speaker verification + human identity, accent A/B, naturalness listening.

## 5.3 Phase 2 — Generalization & Quality
Initial held-out target: ~5–10 speakers / 100–300 utterances.
Categories: scripted BPO, critical entities, pronunciation contrasts, already-correct speech, spontaneous speech.
Measure correction and damage separately.
Telephony sanity: 16 kHz → 8 kHz → G.711, optionally controlled call-center noise.
Pass only if intended pronunciation improves on unseen speakers without unacceptable content/identity damage.

## 5.4 Phase 3 — True Streaming
state = model.create_state()
output, state = model.process_chunk(audio_chunk, state)
Bounded state and bounded lookahead; no whole utterance requirement.
No reference speaker in production; no text/LLM critical path.
Measure algorithmic latency, compute latency, first-output latency, RTF, backlog, state size, memory.
Chunk sweep: 160 / 80 / 40 / 20 ms. Lookahead sweep: 0 / 20 / 40 / 80 / 160 ms where needed.
Verify causality: future audio outside declared lookahead cannot change already-emitted output.
Long-stream tests: 30 min, then 1–2 hours for viable candidates. Check memory/state growth, latency drift, backlog, timing drift, NaNs, silence/restart behavior, artifacts.
Authoritative provisional pilot objective: P95 added latency ≤250–300 ms and RTF ≤0.6; stronger later commercial objective P95 ≤200 ms. Earlier stricter 0.25/200ms numbers in this conversation are aspirational, not Phase-0 gates.

## 5.5 Phase 4 — CPU / endpoint runtime
Source-of-truth target: Windows 10/11 x86-64, 8–16 GB RAM, Intel/AMD CPU, no dedicated GPU requirement.
Optimize only after quality works: ONNX export, ONNX Runtime, operator fusion, quantization, distillation, smaller encoder, lighter synthesis, thread tuning.
Profile exact CPU, threads, RAM, OS, P50/P95/P99, RTF, CPU%, memory, model load and first-output latency.
Native runtime uses bounded input/output rings; no unbounded queues.
Fail-open: ACTIVE → DEGRADED → CROSSFADE → BYPASS; recovery uses hysteresis.
Pass continuous real-time conversion with bounded latency/memory/no backlog/fail-open for at least 30–60 minutes before packaging.

## 5.6 Phase 5 — Pilot-ready MVP
Simple agent UI: status, input mic, target, strength, enable/disable, diagnostics.
Expose “AccentEdge Microphone” to OS/softphones.
No deep CCaaS integration required for MVP.
Diagnostics: model readiness, latency, CPU/memory, bypass, runtime/model version.
Metadata-only logs by default; no recorded call audio.
Installer/uninstaller/runtime/model/virtual audio component/config; no terminal commands.
Reliability tests: 2h call simulation, headset disconnect/reconnect, softphone restart, CPU overload, model error, audio device change, bypass/recovery.

## 5.7 Phase 6 — Controlled BPO Pilot


## 6. Legacy Engineering — What Was Built, What Failed, What Survives
*(27 paragraphs, 234 words)*


## 6.1 Seed-VC prototype — verified legacy facts
A forensic audit dated 25 Aug 2026 found a genuine working offline Seed-VC integration, but explicitly not a real-time/streaming accent product.

## 6.2 Seed-VC module details worth retaining
Actual call path: CLI → ConversionEngine → SeedVCConverter → semantic content/ref features → CAMPPlus reference style → RMVPE F0 → CFM inference → BigVGAN → WAV.
Legacy capture default: 16 kHz, ~200 ms chunks, float32 mono.
Legacy queues were unbounded in capture/playback; this is explicitly incompatible with real-time backpressure safety.
RingBuffer itself was fixed-capacity/thread-safe but allocated on reads and held locks during copies.
Silero VAD wrapper existed but real model loading was not exercised in the live pipeline.
Sample-rate mismatch existed: capture/playback 16 kHz while Seed-VC model path used ~22.05 kHz, requiring resampling both ways.
Dependencies were largely unpinned; model files had no hash/version pinning; torch.load checkpoint supply-chain risk existed.

## 6.3 Legacy bugs/risks recorded

## 6.4 FACodec branch — corrections and current status

## 6.5 Critical FACodec timbre-path lesson

## 6.6 What legacy code may still be reused
Streaming session abstractions.
Causality harness.
Latency profiler and state-growth tests.
Chunk/lookahead sweep tooling.
Experiment/manifest logic.
Some simple baseline/Candidate D code if clean and relevant.

## 6.7 What is retired as evidence
Homemade speaker-similarity metric.
Mel L1 labeled as content preservation.
Griffin-Lim-based naturalness/identity claims.
RTF 0.0044-style product-performance claims.
Stage A speaker-preservation claims.
“Phase 2 complete” and “Candidate D winner” labels.


## 7. Model Landscape — Local Audio-to-Audio, Accent and Real-Time Candidates
*(29 paragraphs, 351 words)*

The thread explored two distinct categories: (A) models that explicitly target accent normalization but are not necessarily production streaming engines, and (B) real-time voice-conversion engines that may provide a shortcut if they can be trained on same-speaker cross-accent pairs.

## 7.1 CosyAccent details
Official ICASSP 2026 implementation; Hugging Face card identifies MIT license.
Direct WAV→WAV inference.
Default `--reference_wav` can fall back to source WAV for timbre conditioning.
Default `--n_timesteps` = 32 flow-matching steps in current repo, making it a quality/teacher candidate rather than assumed streaming endpoint.
Other exposed knobs include duration preservation and speech length ratio.
Use first to establish how good an accent-specific teacher can sound, not as proof of endpoint feasibility.
CosyAccent GitHub: https://github.com/P1ping/CosyAccent
CosyAccent weights: https://huggingface.co/Piping/CosyAccent — MIT metadata; ICASSP 2026 model card.
CosyAccent paper: https://arxiv.org/abs/2602.19166v1

## 7.2 TokAN-Legacy details
Official Interspeech 2025 legacy implementation for Accent Normalization Using Self-Supervised Discrete Tokens.
Repository exposes training steps, including token-to-token pretraining/fine-tuning and use of LibriTTS-R + L2-ARCTIC/ARCTIC in the research recipe.
Long-audio inference splits by Silero VAD; this is chunked offline inference, not automatically a bounded-lookahead live-call architecture.
Useful because it exposes an actual accent-learning pipeline rather than only a pretrained black-box inference script.
TokAN-Legacy: https://github.com/P1ping/TokAN-Legacy — MIT repository; newer TokAN repo exists, but Legacy was the clearer training reference in the discussion.

## 7.3 Real-time engines: what they solve vs what they do not
GENERIC VOICE CONVERSION
Speaker A + reference/target voice → speech that sounds like target speaker

ACCENTEDGE REQUIREMENT
Speaker A, Indian English → Speaker A, US-oriented pronunciation
identity SAME • words SAME • emotion SAME • accent/pronunciation DIFFERENT

## 7.4 Real-time lab stack recommendation
RVC: fastest low-friction experiment; official project MIT and widely used for real-time VC.
Beatrice: important low-latency candidate; keep engine/data licenses separate.
Seed-VC: benchmark its zero-shot/realtime behavior but avoid committing a proprietary distribution strategy to GPL until licensing strategy is explicit.
w-okada: use as a laboratory to switch RVC/Beatrice/DDSP/SVC backends and prove conversational behavior before writing custom device/runtime code.
RVC official: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
Beatrice VST: https://github.com/prj-beatrice/beatrice-vst
Seed-VC: https://github.com/Plachtaa/seed-vc
w-okada voice-changer: https://github.com/w-okada/voice-changer
w-okada license notice: https://github.com/w-okada/voice-changer/blob/master/LICENSE-NOTICE — Shows backend/vocoder/Beatrice data licenses must be audited separately.


## 8. Existing Real-Time Model Shortcut Strategy
*(21 paragraphs, 236 words)*


## 8.1 Model-pluggable runtime
AccentEdge Runtime
       │
       ▼
IVoiceEngine / IAccentEngine
       │
   ┌───┼───────────────┬──────────────┐
   ▼   ▼               ▼              ▼
 RVC  Beatrice      Seed-VC      AccentEdge proprietary
   │   │               │              │
   └───┴───────────────┴──────────────┘
                  ↓
           same streaming/audio API
class IVoiceEngine {
  virtual bool loadModel(path) = 0;
  virtual AudioFrame process(AudioFrame in) = 0;
  virtual void reset() = 0;
  virtual float latencyMs() = 0;
  virtual float realtimeFactor() = 0;
};

## 8.2 The key experiment
Train a real-time VC/student engine not as Speaker A→Speaker B, but as the same speaker across two accent conditions:
SOURCE: Speaker A — natural Indian English
TARGET: Speaker A — coached/natural US-oriented English

Train RVC / Beatrice / another realtime student on matched pairs.

## 8.3 Synthetic teacher shortcut
Permissively licensed Indian-English source speech
                ↓
     high-quality offline teacher
    (CosyAccent / validated strategy)
                ↓
 same-speaker synthetic normalized target
                ↓
      paired synthetic corpus
                ↓
  real-time student (RVC/Beatrice/etc.)
                ↓
  fine-tune/calibrate on human paired data
This is only allowed if the teacher-generated target passes the Phase-0 identity/content/accent gates and the teacher/checkpoint/data licenses permit use for training/target generation. A non-commercial or inference-only teacher cannot be assumed safe for commercial distillation.

## 8.4 Five experiments previously proposed

## 8.5 Scorecard for every real-time candidate
Accent strength / per-dimension correction.
Damage to already-correct phones.
Speaker similarity + human same-person judgment.
WER/CER + critical-entity accuracy.
Naturalness and emotion preservation.
First-output latency, P50/P95 compute, algorithmic lookahead, RTF.
CPU/GPU/RAM/VRAM.
Long-stream stability.
Windows/macOS/Linux portability.
Code, weights, dataset, vocoder, and dependency licenses.


## 9. Proprietary Model Architecture — Only If Existing Real-Time Engines Fail
*(13 paragraphs, 237 words)*

The thread developed a detailed FACodec-style architecture. It remains useful as Plan B / long-term moat, but current source-of-truth says not to build this before the valid-target gate and simple baseline experiments.

## 9.1 Conceptual architecture
Speech waveform
      ↓
Factorized speech encoder
      ├── Content ───────┐
      ├── Prosody ───────┼──► Causal Accent Mapper / Conformer ───┐
      ├── Timbre ───────────────────────────────────────────────────┤
      └── Acoustic residual/detail ──────────────────────────────────┤
                                                                    ▼
                                                               Decoder
                                                                    ↓
                                                             output speech

## 9.2 Why causal Conformer / feed-forward student was preferred
A flow-matching/diffusion teacher may need many iterative steps; that is difficult to push to CPU live-call latency.
A causal/bounded-context Conformer or similarly efficient sequence model can emit incrementally in one feed-forward path per chunk.
The production student should be quantizable and exportable to ONNX/other native runtimes.
Distillation allows a high-quality offline teacher to supervise a smaller causal student.

## 9.3 Speaker state idea (exploratory)
Instead of estimating speaker identity independently every 160–320 ms, maintain a rolling speaker state/embedding across the session. Update conservatively (e.g., EMA) and condition the decoder consistently. This remains exploratory because the speaker representation itself can leak accent; any implementation must be tested against the gold-calibrated identity budget.

## 9.4 Conversion strength as one control surface
INITIALIZING → SOFT_CONVERSION → STABLE_CONVERSION
                      ↑                    ↓
                CROSSFADE_UP ← BYPASS ← CROSSFADE_DOWN ← DEGRADED
The same continuous strength parameter can govern warm-up, the correction-vs-damage operating point, and fail-open ramp-down. The model should therefore accept continuous strength conditioning rather than only binary on/off.


## 10. Streaming Audio & Runtime Architecture
*(13 paragraphs, 166 words)*


## 10.1 Frame and internal audio assumptions discussed

## 10.2 Preprocessing pipeline proposed
Physical mic
  ↓
DC removal / basic conditioning
  ↓
resample
  ↓
noise suppression (if needed)
  ↓
AEC where system topology requires it
  ↓
gain normalization
  ↓
VAD
  ↓
Accent model
  ↓
post-DSP / overlap / crossfade
  ↓
virtual microphone

## 10.3 VAD / noise components discussed
Silero VAD: small, widely used, MIT; already appeared in legacy TokAN/AccentEdge workflows.
RNNoise: permissive BSD-3-Clause neural-noise-suppression baseline; do not waste early R&D inventing custom denoising.
Noise/AEC should not mask core pronunciation feasibility. Keep Phase-0 clean first; add NB/NOISY degradation deliberately.
Silero VAD: https://github.com/snakers4/silero-vad
RNNoise: https://github.com/xiph/rnnoise

## 10.4 Real-time metrics — definitions

## 10.5 Earlier performance heuristics — preserved but downgraded
Earlier in the thread, the following intuitive RTF examples were used to explain concurrency. They remain useful intuition, not capacity commitments:
Do not translate these into “one L4 handles N calls” until the actual model is benchmarked with batching, I/O, memory, p95 latency, and quality under concurrency.


## 11. Cross-Platform Desktop Architecture — Windows First, Not Windows Only
*(22 paragraphs, 327 words)*


## 11.1 Shared core vs OS adapters
                    AccentEdge Core (C++20)
        DSP • VAD • streaming • session state • model API
                           │
                 platform abstraction
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
     Windows             macOS              Linux
     WASAPI              CoreAudio          PipeWire
     virtual mic         HAL plug-in        virtual source
     WinML/TRT/CPU       CoreML/ANE/CPU      TRT/OpenVINO/CPU

## 11.2 Important macOS correction
The discussion corrected an earlier AudioDriverKit idea: Apple’s current documentation says virtual audio devices should use an Audio Server Driver Plug-in; AudioDriverKit is intended for physical devices. macOS therefore uses CoreAudio/HAL virtual device concepts, while inference can use ONNX Runtime CoreML / Apple hardware.
Apple virtual audio sample: https://developer.apple.com/documentation/coreaudio/creating-an-audio-server-driver-plug-in

## 11.3 Linux advantage
PipeWire can create virtual sources/sinks and loopback streams without a custom kernel driver for the common case, which makes Linux attractive for thin clients, VDI, kiosks, and the on-prem inference server.
PipeWire loopback: https://docs.pipewire.org/1.2/page_module_loopback.html

## 11.4 Windows integration
Microsoft’s SysVAD sample demonstrates a WDM virtual audio device and WaveRT architecture. The kernel/device layer should stay intentionally simple: no PyTorch, CUDA, ONNX graph, or complex neural DSP inside the driver. Neural inference belongs in a user-space service.
Microsoft SysVAD sample: https://learn.microsoft.com/en-us/samples/microsoft/windows-driver-samples/sysvad-virtual-audio-device-driver-sample/
SysVAD GitHub mirror: https://github.com/microsoft/audio/blob/main/Samples/Audio/sysvad/README.md

## 11.5 Shared desktop UI choice
The thread initially proposed C# + WinUI 3 for a Windows-only agent UI. After the cross-platform question, the preferred shared UI became Flutter desktop with a C ABI/FFI into the C++ engine. Keep all timing-sensitive audio/ML work in C++; the UI remains a thin control surface.
Flutter UI
    │ FFI / C ABI
    ▼
AccentEdge C++ SDK
    ├── Windows audio backend
    ├── macOS audio backend
    └── Linux audio backend

## 11.6 Architecture interfaces
AccentEngine
IAudioInput
IAudioOutput
IVirtualMicrophone
IInferenceBackend

IInferenceBackend implementations:
  OrtCpuBackend
  TensorRTBackend
  CoreMLBackend
  OpenVINOBackend
  MIGraphXBackend / other supported EP

## 11.7 CPU architecture portability
Windows x86-64 first commercial target.
Linux x86-64 and ARM64 should not be blocked by core assumptions.
macOS ARM64 is first-class; Intel macOS can be lower priority.
Future portability could include Jetson, ARM thin clients, Qualcomm Windows, Android/iPad only if customer demand exists.


## 12. On-Prem / Production Hardware & Deployment
*(20 paragraphs, 199 words)*


## 12.1 Deployment modes discussed

## 12.2 Hardware sizing tiers discussed

## 12.3 Proposed first on-prem appliance
CPU       AMD EPYC / Intel Xeon, ~24–32 physical cores
GPU       2 × NVIDIA L4 24GB (only after single-L4 benchmarking justifies it)
RAM       128GB ECC
OS        Ubuntu Server 24.04 LTS
Storage   redundant OS NVMe + 2–4TB model/cache/log NVMe
Network   dual 10GbE preferred
Runtime   Docker + NVIDIA Container Toolkit + model runtime
Services  audio gateway → VAD/preprocess → accent model → decode/post-DSP → telephony/WebRTC

## 12.4 Hardware facts used in discussion
NVIDIA L4: https://www.nvidia.com/en-in/data-center/l4/
NVIDIA L40S: https://www.nvidia.com/en-in/data-center/l40s/
RTX PRO 6000 Blackwell Server Edition: https://www.nvidia.com/en-in/data-center/rtx-pro-6000-blackwell-server-edition/

## 12.5 Never buy capacity before measuring these
VRAM and RAM per stream/model instance.
Model load time.
First-output and steady-state latency.
RTF under realistic chunk/lookahead configuration.
Max concurrent streams at p95 latency target.
GPU utilization and batching effects.
WER/critical-entity/speaker/accent quality as concurrency rises.
Long-stream degradation, memory growth, underruns/overruns.

## 12.6 High-availability preference
When production requires HA, prefer two independent inference nodes before simply stacking GPUs into one server. Two GPUs in one chassis still share motherboard, OS, power path, and failure domain.
Load balancer / session router
        │
   ┌────┴─────┐
   ▼          ▼
Node A       Node B
1–2×L4      1–2×L4
   │          │
   └────metrics/health────


## 13. Frameworks, Languages & Production Tech Stack
*(17 paragraphs, 211 words)*


## 13.1 ONNX Runtime hardware abstraction
ONNX Runtime’s execution-provider architecture was selected because the same ONNX graph can be routed through CPU, NVIDIA TensorRT/CUDA, Intel OpenVINO, Apple CoreML, AMD/other providers depending target hardware.
ONNX Runtime execution providers: https://onnxruntime.ai/docs/execution-providers/
ONNX Runtime TensorRT: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html
ONNX Runtime CoreML: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html
ONNX Runtime OpenVINO: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html

## 13.2 Endpoint runtime safety model
Capture Thread
     ↓
Bounded Input Ring
     ↓
Inference Worker
     ↓
Bounded Output Ring
     ↓
Virtual Microphone

NO unbounded queues.
If inference falls behind → degrade/crossfade/bypass; never accumulate seconds of audio.

## 13.3 Proposed model update chain
signed manifest + model hash + version + minimum runtime
          ↓
download model B
          ↓
verify signature/checksum
          ↓
load + health check B
          ↓
atomic switch A → B
          ↓ failure
rollback to A

## 13.4 Model manifest fields discussed
{
  "model_version": "1.3.7",
  "accent": "en-US-neutral",
  "sample_rate": 16000,
  "quantization": "int8",
  "sha256": "...",
  "minimum_runtime": "1.4.0"
}

## 13.5 Control-plane data model discussed
organizations
sites
users
devices
licenses
models
model_versions
deployments
sessions
usage_metrics
audit_events

## 13.6 Observability metrics discussed
accent_frame_latency_ms
accent_rtf
audio_underruns / audio_overruns
model_load_ms
vad_speech_ratio
cpu_percent / memory_mb
gpu_percent / gpu_vram_mb
session_duration
driver_errors / model_errors / bypass_events
Raw audio and transcripts should not be logged by default. A diagnostic audio-capture mode, if ever added, needs explicit customer/user policy and retention controls.


## 14. Data Strategy & Dataset Catalogue
*(26 paragraphs, 290 words)*


## 14.1 Dataset-use buckets
PRODUCTION-SAFE CANDIDATES  → only after provenance/license audit
RESEARCH / EVALUATION ONLY → noncommercial or restrictive corpora
LICENSE CLARIFICATION       → commercial rights not implied by public download
PROPRIETARY                 → AccentEdge-collected paired data with explicit consent/rights

## 14.2 Dataset catalogue — free/open candidates
Common Voice Indian Accent HF: https://huggingface.co/datasets/ishands/commonvoice-indian_accent
Kaggle Indian English Accent Audio: https://www.kaggle.com/datasets/kotekalvijay/indian-english-accent-audio
CMU ARCTIC: http://festvox.org/cmu_arctic/
FLEURS: https://huggingface.co/datasets/google/fleurs
LibriTTS-R OpenSLR: https://www.openslr.org/141/
VCTK (HF convenience link): https://huggingface.co/datasets/vt57299/vctk — Verify against original University of Edinburgh corpus terms before production.
IndicVoices: https://huggingface.co/datasets/ai4bharat/IndicVoices
Project Vaani: https://vaani.iisc.ac.in/dataset

## 14.3 Indian-English institutional/commercial sources
LDC-IL portal: https://ldcil.org/
LDC-IL Kannada raw Indian-English corpus: https://data.ldcil.org/indian-english-raw-speech-corpus-kannada-variant
Defined.ai en-IN catalogue: https://defined.ai/datasets?locales=en-IN&pagination_page=1&pagination_pageSize=50
ELRA commercial-use search: https://catalogue.elra.info/en-us/repository/search/?q=&selected_facets=restrictionsOfUseFilter_exact%3ACommercial+Use — Discussed earlier; specific item terms must be re-verified.
ELRA Indian English mobile item (discussed): https://catalog.elra.info/en-us/repository/browse/ELRA-S0456/ — Preserved from earlier scan; re-verify current item page/rights before procurement.

## 14.4 Research-only / restricted datasets
L2-ARCTIC project: https://psi.engr.tamu.edu/l2-arctic-corpus/
Speech Accent Archive: https://accent.gmu.edu/
Speech Accent Archive download/license: https://accent.gmu.edu/download/
Whissle Meta STT EN-IN Tech Interviews: https://huggingface.co/datasets/WhissleAI/Meta_STT_EN-IN_Tech_Interviews

## 14.5 Data lake layout discussed
accentedge-data/
├── production-safe/
│   ├── source-indian/
│   │   ├── commonvoice-en-in/
│   │   ├── kaggle-indian-audit/
│   │   └── ldc-il/
│   ├── target-american/
│   │   ├── cmu-arctic/
│   │   ├── fleurs-en-us/
│   │   └── libritts-r/
│   ├── speaker/
│   │   ├── vctk/
│   │   └── libritts-r/
│   └── indian-pretraining/
│       ├── indicvoices/
│       └── vaani/
├── research-only/
│   ├── l2-arctic/
│   └── speech-accent-archive/
└── proprietary/
    └── accentedge-paired/

## 14.6 Required provenance manifest per dataset
dataset name
source URL
original creator / collector
version + download date
license text snapshot + hash
speaker consent / release status
commercial model-training right
target-generation/distillation right
derivative model/weight distribution right
redistribution right
voice cloning / biometric restrictions if any
file checksums
allowed use category: EVAL_ONLY / TRAIN / TARGET_GEN / PROD


## 15. Proprietary AccentEdge-PAIR Dataset Strategy
*(13 paragraphs, 210 words)*


## 15.1 Why this becomes the moat
Open corpora supply Indian-English source speech and native-US speech, but do not give a large commercially clean dataset where the same speaker says the same sentence in a natural Indian-English condition and a US-oriented condition. That axis is exactly what the product must learn and what identity metrics must calibrate against.

## 15.2 Do not jump straight to 50 speakers

## 15.3 Long-term collection plan discussed

## 15.4 Recording contract
Same human, same sentence/script in both conditions.
Natural Indian-English condition first; target condition should be natural/code-switched/coached without changing persona unnecessarily.
Same microphone/headset, room, sampling format, and approximate emotional intent.
Capture natural/spontaneous speech, not only studio read sentences.
Recruit across Hindi/Punjabi/Bengali/Gujarati/Marathi/Tamil/Telugu/Kannada/Malayalam and other substrate backgrounds rather than treating “Indian accent” as one class.
Obtain explicit informed consent for ML training, derived model weights, commercial deployment, and biometric/voice processing where applicable.

## 15.5 Synthetic paired-data scale-up
Potential shortcut after the teacher is validated: take a commercially clean Indian-English corpus (e.g., Common Voice Indian subset), generate same-speaker normalized teacher targets, then train a real-time student. The earlier discussion called this “~164 hours paired almost immediately,” but the master downgrades that to a potential capacity estimate: quality, licenses, speaker identity and teacher rights must be proven first.


## 16. Evaluation, Benchmarks & Production Gates
*(41 paragraphs, 230 words)*


## 16.1 Content safety is first-class
WER and CER with one fixed ASR evaluator across conditions.
Human critical-entity transcription, not ASR alone.
Numbers: fifteen/fifty, thirteen/thirty, amounts, dates, appointment times.
Names, addresses, account/claim IDs, alphanumeric codes.
Semantic-equivalence scoring where format differs but meaning is identical (e.g. “fifteen dollars” vs “$15”).

## 16.2 Identity evaluation
Established speaker verification, not homemade mel statistics.
At least ECAPA-TDNN + WavLM-based verification head + one independent verifier in the calibration protocol.
Normalize each metric against same-session, different-session, style-shift, natural cross-accent, and imposter conditions.
Human same-person judgment remains decisive when objective encoders disagree.

## 16.3 Accent/pronunciation evaluation
Human A/B listening early.
Per-dimension pronunciation probes later.
Correction over DEVIANT tokens; damage over ALREADY-TARGET tokens.
Off-target movement tracked separately.
One accent classifier may be secondary diagnostics, never the headline metric.

## 16.4 Naturalness & emotion
5-point MOS with hidden natural anchors.
Emotion/intended-style match plus confidence.
Free-text identity feedback to detect “sounds like them doing an impression” or “another person wearing their voice.”

## 16.5 Telephony and realistic acoustic tests
16kHz original.
8kHz downsample.
G.711 μ-law round-trip.
Call-center babble/noise around the protocol’s controlled SNR.
Headset-response proxy.
NB+NOISY combined.

## 16.6 Production engineering gates discussed — status distinction

## 16.7 Long-stream failure checks
Memory/state growth.
Latency drift.
Backlog.
Output timing drift.
NaNs or model-state corruption.
Audio artifacts at boundaries.
Silence/restart behavior.
Headset disconnect/reconnect.
CPU overload.
Device changes.
Bypass and hysteretic recovery.


## 17. Licensing, Provenance, Privacy & Enterprise Security
*(26 paragraphs, 350 words)*


## 17.1 Four separate license layers
1. SOURCE CODE LICENSE
2. PRETRAINED WEIGHTS / CHECKPOINT LICENSE
3. TRAINING / TARGET-GENERATION DATA LICENSE
4. BUNDLED DEPENDENCIES / VOCODERS / SPEAKER MODELS / SUBMODULES
An MIT GitHub repository does not automatically make every downloaded voice, checkpoint, dataset, teacher-generated target, or bundled vocoder commercially safe.

## 17.2 Model-specific license lessons from the scan
CosyAccent HF model card: MIT; still audit dependencies and source-synthesis training data rights for your intended use.
TokAN-Legacy repo: MIT; its reference training recipe includes L2-ARCTIC, which is noncommercial — do not copy a recipe into commercial training without replacing/relicensing the data.
FACodec current HF checkpoint metadata: Apache-2.0; record exact model revision and all dependencies.
Vevo weights: CC BY-NC 4.0; study architecture but do not use weights commercially without permission.
Seed-VC: GPL-3.0; commercial use is not “banned,” but copyleft obligations can conflict with a proprietary closed-source distribution plan.
Beatrice engine: MIT; JVS/bundled voice assets have separate restrictions.
w-okada: license notice explicitly points to backend/vocoder/Beatrice-specific terms.
Whissle: inference-only license explicitly forbids training/distillation.

## 17.3 Security architecture discussed
No raw call audio to cloud by default.
mTLS for agent↔on-prem gateway streaming if server mode is used.
Signed model manifests + SHA-256; pin model revisions/checkpoints.
Do not blindly torch.load untrusted checkpoints in production; convert/sign internally controlled artifacts.
Metadata-only telemetry by default.
Per-device licensing, runtime/model version reporting, health/bypass events.
Fail-open to original microphone if inference fails.
No unbounded queues that turn transient overload into ever-growing call delay.

## 17.4 Enterprise privacy advantage
A local-first endpoint mirrors the strongest part of the competitive proof: Sanas currently emphasizes on-device real-time processing and that conversation data/voice samples are not stored. For BPOs serving banking, healthcare, telecom and other regulated clients, “audio remains on endpoint/customer network” materially simplifies architecture/security review.
Sanas privacy-first Accent Translation docs: https://help.sanas.ai/docs/accent-translation

## 17.5 Voice-data consent
Because speaker identity is biometric/voice data, proprietary data collection should use an explicit contributor release covering recording, preprocessing, model training, commercial derived weights, internal evaluation, retention/deletion, and any permitted voice/speaker-embedding use. This master is an engineering/business record, not legal advice; counsel should review final licenses and consent language.


## 18. Business Economics, Competitive Proof & Go-to-Market
*(24 paragraphs, 196 words)*


## 18.1 Competitive proof that the category exists
Sanas markets real-time Accent Translation to enterprises/contact centers.
Current product docs list source accents including India and targets including the US/UK.
Product docs say personalized voice characteristics are retained and processing is on-device in real time.
Public AWS Marketplace pricing gives a concrete per-seat benchmark.
Sanas documentation also shows endpoint/headset/system requirements and a virtual-audio style workflow in the calling app.
Sanas system requirements: https://help.sanas.ai/docs/sanas-system-requirements — Current Aug 2026 system-requirement documentation can be used as an external endpoint-compute sanity benchmark.

## 18.2 Positioning options discussed

## 18.3 Sales proof to collect in pilot
Does listener clarification/repetition decrease?
Does AHT move without harming QA/FCR?
Do agents voluntarily keep it enabled?
Does agent fatigue or self-consciousness increase?
Does it ever mutate high-risk entities?
Can IT deploy it without changing CCaaS stack?
Does privacy/local-processing reduce procurement friction?
What price/seat would customer actually extend after pilot?

## 18.4 What not to claim before data
Guaranteed CSAT/AHT/FCR improvement.
“No latency” or an exact millisecond number without device/model benchmark.
Exact calls-per-GPU capacity.
“Removes accents” universally.
Commercial safety of an open-source model because the code repository is permissive.
Same-speaker preservation based only on one embedding score.


## 19. Master Risk Register & Unresolved Questions
*(9 paragraphs, 89 words)*


## 19.1 Key unresolved architecture choices
Which teacher strategy clears TGFP v2: CosyAccent, Strategy B TTS→identity, Strategy C sparse, another teacher?
Can Beatrice/RVC learn same-speaker accent mapping with paired supervision?
If not, is FACodec-style factorization actually the right proprietary representation?
What exact streaming chunk/lookahead gives quality without >200–300ms p95?
Can final student run CPU-only on real BPO endpoint hardware?
Does BPO need on-prem GPU gateway as permanent product or only bridge?
Which datasets have explicit rights for commercial target generation/distillation?
What is the first validated price/seat and purchase motion?


## 20. Immediate Execution Plan & Decision Tree
*(14 paragraphs, 225 words)*


## 20.1 Current sprint: VALID TARGET
Resolve every legacy Stage-A pair: corpus, source speaker, target speaker, same transcript, same human, model, synthesis path.
Retire the old homemade speaker metric as evidence; document exactly what it computed.
Record current repo integrity: git commit/status/tests for reproducibility, without treating tests as scientific success.
Ensure synthesis/reconstruction path is competent and upstream-faithful.
Use one established speaker-verification model plus human identity.
Use a fixed ASR for WER/CER plus manual critical entities.
Produce one same-speaker target.
Listen blind enough to answer same person / same words / desired shift / natural enough.
Only if convincing, expand to 10–30 high-quality targets and run the TGFP v2 teacher strategies.
Only after trustworthy targets, run a tiny-overfit baseline / existing real-time student experiment.

## 20.2 Decision tree after Phase-0 evidence
VALID SAME-SPEAKER TARGET?
   ├── NO → fix target generation / supervision; do NOT build streaming runtime
   └── YES
       ↓
CAN EXISTING REAL-TIME ENGINE (RVC/Beatrice/etc.) LEARN IT?
   ├── YES → benchmark generalization + streaming → optimize/export → MVP
   └── NO
       ↓
CAN SIMPLE BASELINE LEARN IT OFFLINE?
   ├── NO → revisit representation/target/loss
   └── YES
       ↓
FAILURE-DRIVEN ARCHITECTURE
   ├── context bottleneck → streaming Conformer/Emformer ideas
   ├── synthesis bottleneck → lighter vocoder/DDSP ideas
   ├── pronunciation mapping → token/phonetic representation
   └── whole-speech identity failure but sparse repair works → sparse-repair product branch

## 20.3 The seven milestones that matter


## Appendix A — Exact Phase-0 Test Material Preserved from TGFP v2
*(26 paragraphs, 313 words)*


## A.1 Set A — Scripted BPO (10)
Thank you for calling. My name is Priya. How may I help you today?
Could you please confirm your account number and the billing address on file?
I can see a charge of thirty dollars posted on the thirteenth of August.
Let me check the details and get back to you in about fifteen minutes.
The total amount due is one hundred and forty-four dollars.
Unfortunately, that particular service is not available in your area at the moment.
Your appointment has been rescheduled for Thursday the third at four thirty.
Can you spell that for me? A as in Alpha, B as in Bravo, seven, one, three, nine, K.
I understand your frustration, and I want to sort this out for you right away.
I'll transfer you to our technical department. Please hold for a moment.

## A.2 Set B — Contrast-dense diagnostic (3)
The water in the quarter-litre bottle was better than the third one we ordered.
We will verify whether the vendor’s development schedule is available for review.
She thought the three tickets were worth thirty pounds, but they cost thirty-three.

## A.3 Set C — Spontaneous (3 prompts)
Describe your commute for 20–30 seconds.
Explain a process you know well for 20–30 seconds.
Recount a mildly annoying recent experience for 20–30 seconds.
Do not omit spontaneous speech. It exposes disfluency, speaking rate, coarticulation, reduced articulation, and alignment failure that scripted speech can hide.

## A.4 Set D — Already-target-dense (4)
She placed a simple order for six items on Friday morning.
Please send me the file before the meeting starts.
The system is running slowly, so I’ll restart the machine.
My colleague will call you back after lunch to finish the process.
These are candidate already-target-dense sentences, not assumptions. Per-token realization labels are speaker-specific and must determine which tokens are actually ALREADY-TARGET.


## Appendix B — Repository, Service & Interface Blueprints
*(9 paragraphs, 188 words)*


## B.1 Fresh-start research repository (source-of-truth version)
accentedge/
├── docs/
│   ├── PROJECT_CONTEXT.md
│   ├── PHASE0.md
│   └── decisions/
├── experiments/
├── data/
│   ├── manifests/
│   └── README.md
├── src/accentedge/
│   ├── target_generation/
│   ├── model/
│   ├── evaluation/
│   ├── streaming/
│   └── runtime/
├── tests/
└── legacy/
    └── README.md

## B.2 Later production monorepo concept discussed
accentedge/
├── model-lab/
├── core/
│   ├── inference/
│   ├── dsp/
│   ├── streaming/
│   ├── vad/
│   ├── speaker/
│   └── include/
├── platform/
│   ├── windows/{wasapi,virtual-mic,service}/
│   ├── macos/{coreaudio,hal-plugin,service}/
│   └── linux/{pipewire,service}/
├── desktop/flutter/
├── inference-server/
├── control-plane/
├── dashboard/
├── infra/
├── benchmarks/
└── docs/
The second layout is POST-GATE. The first layout is deliberately simpler and should be used until Phase 0/1 evidence justifies production infrastructure.

## B.3 On-prem gRPC contract concept
StartSession
PushAudioFrame
ReceiveAudioFrame
UpdateConfig
EndSession

Transport: bidirectional gRPC streaming + mTLS
Audio: PCM16, 16 kHz, mono, ~20 ms packets
Do not Base64/JSON-encode raw audio.

## B.4 Model engine contract concept
initialize(ModelConfig)
process(AudioFrame)
setAccent(AccentProfile)
setStrength(float)
reset()
metrics()

The engine must not know whether it is called from Windows, macOS, Linux, or an on-prem server.


## Appendix C — Chronological Decision & Discussion Log
*(0 paragraphs, 0 words)*




## Appendix D — Complete Link Register
*(77 paragraphs, 874 words)*

Links below include sources explicitly discussed in this thread plus official references used to verify the current technical stack. “Verify exact asset” means the URL is useful but the specific checkpoint/data/version license must still be recorded in the project manifest before commercial use.
MODELS — CosyAccent GitHub: https://github.com/P1ping/CosyAccent — Official ICASSP 2026 implementation.
MODELS — CosyAccent HF: https://huggingface.co/Piping/CosyAccent — MIT model card; model weights.
MODELS — CosyAccent paper: https://arxiv.org/abs/2602.19166v1 — Duration-controllable accent normalization.
MODELS — TokAN-Legacy: https://github.com/P1ping/TokAN-Legacy — MIT; full accent-normalization training reference.
MODELS — FACodec checkpoint: https://huggingface.co/amphion/naturalspeech3_facodec — Current model card Apache-2.0.
MODELS — Amphion: https://github.com/open-mmlab/Amphion — Speech/audio generation toolkit containing FACodec/Vevo integrations.
MODELS — seq2seq-vc: https://github.com/unilight/seq2seq-vc — MIT; includes L2-ARCTIC foreign-accent-conversion recipe.
MODELS — PPG2PPG: https://github.com/warisqr007/ppg2ppg — Accent conversion research pipeline; verify all assets.
MODELS — CosyVoice: https://github.com/QwenAudio/CosyVoice — Apache-2.0 repo; VC/training capabilities.
MODELS — SpeechT5 VC: https://huggingface.co/microsoft/speecht5_vc — MIT audio-to-audio model card.
MODELS — RVC official: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI — MIT official project.
MODELS — OpenVoice: https://github.com/myshell-ai/OpenVoice — Official repo; V2 commercial-use discussion.
MODELS — Seed-VC: https://github.com/Plachtaa/seed-vc — GPL-3.0; real-time/zero-shot VC reference.
MODELS — Vevo HF: https://huggingface.co/amphion/Vevo — CC BY-NC 4.0 released weights; accent/style conversion.
MODELS — Vevo Amphion README: https://github.com/open-mmlab/Amphion/blob/main/models/vc/vevo/README.md — Architecture/usage reference.
MODELS — Beatrice VST: https://github.com/prj-beatrice/beatrice-vst — MIT real-time voice conversion DSP engine.
MODELS — Project Beatrice V2 org: https://github.com/Project-Beatrice-V2/.github — Community training/packaging ecosystem; audit JVS data terms.
MODELS — w-okada voice-changer: https://github.com/w-okada/voice-changer — Realtime multi-backend laboratory.
MODELS — w-okada license notice: https://github.com/w-okada/voice-changer/blob/master/LICENSE-NOTICE — Backend/vocoder/Beatrice licensing notes.
DATA — Common Voice Indian Accent: https://huggingface.co/datasets/ishands/commonvoice-indian_accent — 163.89h / 110,088 recordings / CC0 card.
DATA — Kaggle Indian English Accent: https://www.kaggle.com/datasets/kotekalvijay/indian-english-accent-audio — 6.01GB / 8,115 files / CC0 page; provenance audit required.
DATA — CMU ARCTIC: http://festvox.org/cmu_arctic/ — Native-US phonetically designed corpus.
DATA — FLEURS: https://huggingface.co/datasets/google/fleurs — Includes en_us; CC BY 4.0 metadata.
DATA — LibriTTS-R: https://www.openslr.org/141/ — Original OpenSLR distribution; verify exact license/version.
DATA — VCTK convenience HF: https://huggingface.co/datasets/vt57299/vctk — Verify original corpus license/source.
DATA — LDC-IL: https://ldcil.org/ — Government of India language resource consortium.
DATA — LDC-IL Kannada Indian-English raw: https://data.ldcil.org/indian-english-raw-speech-corpus-kannada-variant — Commercial/noncommercial user routes.
DATA — IndicVoices: https://huggingface.co/datasets/ai4bharat/IndicVoices — CC BY 4.0 card.
DATA — Project Vaani: https://vaani.iisc.ac.in/dataset — Large Indian speech project.
DATA — Defined.ai en-IN marketplace: https://defined.ai/datasets?locales=en-IN&pagination_page=1&pagination_pageSize=50 — Commercial Indian-English call-center/domain data.
DATA — InfoBayAI HF datasets search: https://huggingface.co/datasets?other=call-center-audio — Use to locate public call-center previews; full rights separately.
DATA — Axon English call-center Kaggle: https://www.kaggle.com/datasets/axondata/english-call-center-speech-recognition — Preview discussed; commercial full data requires separate terms.
DATA — ELRA commercial-use catalogue: https://catalogue.elra.info/en-us/repository/search/?q=&selected_facets=restrictionsOfUseFilter_exact%3ACommercial+Use — Commercial speech-resource catalogue.
DATA — ELRA-S0456 (discussed): https://catalog.elra.info/en-us/repository/browse/ELRA-S0456/ — Re-verify current item and terms.
DATA — L2-ARCTIC: https://psi.engr.tamu.edu/l2-arctic-corpus/ — Research corpus; noncommercial licensing in discussed release.
DATA — Speech Accent Archive: https://accent.gmu.edu/ — CC BY-NC-SA 4.0 current site.
DATA — Speech Accent Archive download: https://accent.gmu.edu/download/ — OSF download/license details.
DATA — Whissle EN-IN Tech Interviews: https://huggingface.co/datasets/WhissleAI/Meta_STT_EN-IN_Tech_Interviews — Inference-only; no training/distillation.
HARDWARE — NVIDIA L4: https://www.nvidia.com/en-in/data-center/l4/ — 24GB / 72W inference candidate.
HARDWARE — NVIDIA L40S: https://www.nvidia.com/en-in/data-center/l40s/ — 48GB ECC / 350W class.
HARDWARE — RTX PRO 6000 Blackwell Server: https://www.nvidia.com/en-in/data-center/rtx-pro-6000-blackwell-server-edition/ — 96GB GDDR7 ECC; high-end R&D/inference.
RUNTIME — ONNX Runtime EPs: https://onnxruntime.ai/docs/execution-providers/ — Cross-hardware execution-provider overview.
RUNTIME — TensorRT EP: https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html — NVIDIA ONNX inference.
RUNTIME — CoreML EP: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html — Apple CPU/GPU/Neural Engine.
RUNTIME — OpenVINO EP: https://onnxruntime.ai/docs/execution-providers/OpenVINO-ExecutionProvider.html — Intel/edge acceleration.
AUDIO — Microsoft SysVAD: https://learn.microsoft.com/en-us/samples/microsoft/windows-driver-samples/sysvad-virtual-audio-device-driver-sample/ — Windows virtual audio device sample.
AUDIO — Windows audio samples: https://learn.microsoft.com/en-us/windows-hardware/drivers/audio/sample-audio-drivers — Microsoft audio-driver references.
AUDIO — Apple Audio Server Driver Plug-in: https://developer.apple.com/documentation/coreaudio/creating-an-audio-server-driver-plug-in — Preferred virtual audio-device mechanism on macOS.
AUDIO — Apple AudioDriverKit note: https://developer.apple.com/documentation/audiodriverkit/creating-an-audio-device-driver — States virtual-device best practice is Audio Server plug-in; AudioDriverKit for physical devices.
AUDIO — PipeWire loopback: https://docs.pipewire.org/1.2/page_module_loopback.html — Virtual sinks/sources and loopback.
AUDIO — Silero VAD: https://github.com/snakers4/silero-vad — VAD.
AUDIO — RNNoise: https://github.com/xiph/rnnoise — BSD neural noise suppression.
FRAMEWORK — PyTorch: https://pytorch.org/ — Training/research.
FRAMEWORK — Hydra: https://github.com/hydra-ecosystem/hydra — Experiment configuration.
FRAMEWORK — MLflow: https://mlflow.org/docs/latest/ml/tracking/ — Experiment tracking.
FRAMEWORK — DVC: https://dvc.org/ — Dataset/model versioning.
FRAMEWORK — FastAPI: https://fastapi.tiangolo.com/ — Control-plane API later.
FRAMEWORK — PostgreSQL: https://www.postgresql.org/ — Control-plane database.
FRAMEWORK — Redis: https://redis.io/ — Ephemeral cache/rate limiting.
FRAMEWORK — Flutter: https://flutter.dev/ — Cross-platform desktop UI proposal.
FRAMEWORK — gRPC: https://grpc.io/ — Bidirectional LAN audio streaming proposal.
OBSERVABILITY — OpenTelemetry: https://opentelemetry.io/ — Telemetry standard.
OBSERVABILITY — Prometheus: https://prometheus.io/ — Metrics.
OBSERVABILITY — Grafana: https://grafana.com/ — Dashboards.
OBSERVABILITY — Loki: https://grafana.com/oss/loki/ — Logs.
INFRA — Docker: https://www.docker.com/ — Server/control-plane packaging.
INFRA — Terraform: https://developer.hashicorp.com/terraform — Infrastructure as code.
BUSINESS — Sanas Accent Translation docs: https://help.sanas.ai/docs/accent-translation — Realtime/on-device competitive proof.
BUSINESS — Sanas app guide: https://help.sanas.ai/v1/docs/using-sanas-app — Device/calling-app workflow.
BUSINESS — Sanas system requirements: https://help.sanas.ai/docs/sanas-system-requirements — Current endpoint hardware requirements.
BUSINESS — Sanas AWS Marketplace: https://aws.amazon.com/marketplace/pp/prodview-mr5luwansb4k6 — Public pricing benchmark.
BUSINESS — Sanas product page: https://www.sanas.ai/accent-translation — Category/enterprise integration proof.
CLOUD REFERENCE — OpenAI GPT-Realtime: https://developers.openai.com/api/docs/models/gpt-realtime — Cloud audio-to-audio reference; not local; fine-tuning currently unsupported.
CLOUD REFERENCE — OpenAI custom voices API: https://platform.openai.com/docs/api-reference/audio/updateVoiceConsent?lang=javascript — Custom voice/consent API reference for eligible customers; not a local model path.

## D.1 Legal/verification disclaimer
Licenses, model cards, dataset availability, prices, system requirements, and vendor terms change. This master records what was discussed and what was re-verified on 27 Aug 2026 where practical. Before commercial training, target generation, redistribution, or deployment, freeze the exact repository/model/data revision and archive the corresponding license/terms. This dossier is not legal advice.


## Final Synthesis — The Architecture to Commit To Without Overbuilding
*(2 paragraphs, 8 words)*

End of master dossier.
As-of 27 August 2026
