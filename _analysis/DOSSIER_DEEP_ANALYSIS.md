# AccentEdge Master Dossier — Complete Deep Analysis

> **Source:** AccentEdge_Master_Dossier_2026-08-27.docx
> **Pages:** 48 | **Paragraphs:** 737 | **Tables:** 62
> **Date:** 2026-08-27
> **Purpose:** Explain the ENTIRE project thesis, architecture, phases, linguistics, business, risks

---

## 1. EXECUTIVE THESIS

**AccentEdge transforms Indian-English speech into US-neutral pronunciation while preserving the speaker's identity, exact words, and emotional intent — for BPO call-centre agents who need American listeners to understand them.**

The document is explicit: this is **NOT** "remove the accent." It is **accent normalization** — a surgical pronunciation adjustment that makes speech more intelligible to US-native listeners while leaving speaker identity, content, and prosody untouched.

---

## 2. WHAT IT IS AND IS NOT

### IS:
- A research project proving identity-preserving accent conversion is achievable
- A BPO-focused product: Indian-English → US-neutral for call-centre agents
- An on-device/local-first system (model runs on the agent's PC, no cloud)
- A 9-phase research program, each gated by scientific evidence

### IS NOT:
- A general-purpose accent removal tool
- A cloud service (explicitly rejected: "$1/minute killed the original model")
- A voice clone or voice changer
- Real-time in the early phases (Phase 0/1 are offline)
- A Windows-only system (cross-platform is the goal)
- Proven to work yet (Phase 0 gates unpassed)

---

## 3. DOCUMENT CONTROL & EVIDENCE HIERARCHY

### Status Labels (Table 1)
| Label | Meaning |
|-------|---------|
| **ACTIVE / SOURCE OF TRUTH** | Wins over older statements |
| **CURRENT DECISION** | Preferred direction, subject to scientific gates |
| **POST-GATE PROPOSAL** | Relevant only after a gate passes |
| **EXPLORATORY** | Candidate discussed for investigation |
| **LEGACY EVIDENCE** | Historical results, not current proof |
| **INVALIDATED** | Prior claims found incorrect |
| **LICENSE CHECK REQUIRED** | Needs legal review before use |

### Evidence Hierarchy (most → least authoritative):
1. This dossier (PROJECT_CONTEXT.md)
2. TGFP v2 (current Phase 0 decision document)
3. Current experimental evidence / artifacts
4. Current code
5. Older audits, reports, milestone labels
6. Old agent summaries

### Claims That Must NOT Be Carried Forward:
The dossier explicitly says these are **NOT proven** and should never be cited as current facts:
- "Phase 2 complete" → Only architecture was evaluated, not implemented
- "Candidate D selected as the winner" → Selection was based on paper analysis, not empirical results
- "Speaker preservation validated" → Never measured at scale
- "Content preservation validated" → Only WER on Whisper, not word-for-word
- "Accent conversion validated" → No human listening test completed
- "Real-time validated" → Only theoretical RTF estimates
- "RTF 0.0044 proves AccentEdge is fast" → Lab-only, not production
- "Stage A proves same-speaker accent conversion" → Stage A used different speakers
- "Benchmark scientifically validated" → Scaffolded, not validated

---

## 4. BUSINESS THESIS & PRICING EVOLUTION

### Original Idea
- Cloud service: $1/minute of audio processed
- Agents upload audio, server processes, returns converted audio
- Revenue: $40/seat/month for 1,000 seats = $40K MRR

### Why It Failed
- **$1/minute is too expensive** for call-centre economics
- **Cloud processing is a liability**: call audio contains PII, can't upload without GDPR/legal risk
- **Competition**: Sanas and others already offer real-time cloud accent conversion
- **Trust**: BPOs don't want their agents' conversations processed by third-party servers

### Current Direction: On-Device / Local-First
- Model runs on the agent's Windows PC — no upload
- BPO pays per-seat licensing fee
- Call audio never leaves the LAN
- Control plane is light: licensing, model distribution, fleet health, analytics
- Hardware: on-prem inference (L4/L40S/RTX PRO tiers)

### Why On-Device Improves Economics
- No per-minute cloud cost after model deployment
- No bandwidth/storage cost for audio
- Privacy is the product, not a cost
- Competitive moat: harder to replicate than a cloud API

---

## 5. LINGUISTIC TRANSFORMATION CONTRACT

### Why "Remove the Indian Accent" Is Wrong
Accent is not an independent knob. It overlaps with:
- Phone realization (how sounds are produced)
- Vowel quality
- Lexical stress
- Duration
- Rhythm
- Intonation

You can't just "turn off" Indian English pronunciation patterns.

### What MUST Change (v1 Scope)
**Segmental (individual sounds):**
- /æ/ raising (cat → [keɪt] pattern)
- /ɪ/ tensing (bit → [bɪt] closer to [bet])
- /oʊ/ monophthongization (go → [goː] instead of [goʊ])
- /uː/ fronting (food → [fʊd] pattern)
- /t/ flapping (water → [wɔɾɚ])
- /θ/ → /t/ substitution (think → [tɪŋk])
- /v/ → /w/ substitution (very → [wɛri])
- /d/ deletion in clusters (hand → [hæn])
- Cluster epenthesis

**Suprasegmental (rhythm/stress):**
- Lexical stress → bounded correction
- Phone duration → allowed to change
- Word duration → bounded empirically
- Phrase timing → prefer preservation

### What MUST Preserve
- **Timbre** (speaker identity — voice print)
- **Voice quality** (breathy, creaky, etc.)
- **Pitch range** (broadly)
- **Emotion** and conversational intent
- **Words exactly** — no content change, no hallucination

### What is OUT of v1 Scope
- Global rhythm changes
- Global intonation changes
- Regional Indian variations (only "standard" Indian English in scope)
- Non-BPO domains (casual speech, singing, etc.)

### The v1 Linguistic Contract
This is the formal specification of what the model must/must-not do. It is the **north star** for all engineering decisions.

---

## 6. TARGET GENERATION FEASIBILITY PROTOCOL (TGFP v2)

### Purpose
A single decisive experiment: **can we construct a usable accent transformation target?** If no, the project stops.

### Gates (−1 through 2)
| Gate | Question | Evidence Required |
|------|----------|-------------------|
| **Gate −1** | Can we annotate pronunciation differences reliably? | Alignment correction rate, AMBIGUOUS count |
| **Gate 0** | Can we produce a target that a human would label as "more US-neutral"? | Human listening test, preference score |
| **Gate 1** | Does the target preserve the speaker's identity? | Speaker similarity metric (ECAPA-TDNN) |
| **Gate 2** | Does the target preserve content exactly? | WER on Whisper, word-level comparison |

### Target Generation Strategies
**Strategy A — Prosodic anchor + local patch**
- Use a US-neutral speaker's prosody as anchor
- Patch in the target speaker's timbre
- Risk: prosody entangles with accent

**Strategy B — Phone-level intervention (PREFERRED)**
- Hand-build target by modifying individual phone realizations
- IPA-level edits to spectrogram
- Most precise but most labor-intensive

**Strategy C — Full synthesis**
- Use TTS to generate target from text
- Risk: loses speaker identity entirely

### Phase-0 Test Material (Appendix A)
**Set A — Scripted BPO (10 sentences):** Standard customer-service dialogues
**Set B — Contrast-dense diagnostic (3 sentences):** Sentences packed with target phonemes
**Set C — Spontaneous (3 prompts):** Free speech prompts to test non-scripted material
**Set D — Already-target-dense (4 sentences):** Sentences where Indian English already sounds close to US-neutral

---

## 7. DELIVERY ROADMAP

### Phases 0–8
| Phase | Name | Duration | Key Milestone |
|-------|------|----------|---------------|
| 0 | Target Feasibility | ~10 weeks | TGFP v2 passes all 4 gates |
| 1 | BPO Benchmark | ~8 weeks | 10-20 speakers, benchmark validated |
| 2 | Architecture Bake-off | ~6 weeks | Candidate D selected (already done) |
| 3 | AccentEdge S2S Model | ~12 weeks | First proprietary model trained |
| 4 | Streaming Inference | ~8 weeks | Real-time chunked conversion |
| 5 | Optimisation | ~6 weeks | Latency < 200ms, RTF < 0.1 |
| 6 | Runtime | ~8 weeks | Windows endpoint, admin portal |
| 7 | Pilot | ~6 weeks | BPO deployment with 10-50 seats |
| 8 | Production | Ongoing | Full rollout, continuous improvement |

### The Seven Milestones That Matter
1. TGFP v2 passes all 4 gates
2. Benchmark validated on 10+ speakers
3. Phase 3 model trained and evaluated
4. Streaming inference at < 200ms latency
5. Windows runtime packaged and tested
6. Pilot with real BPO agents
7. Production deployment

### Decision Tree After Phase 0
```
Phase 0 passes all gates?
├── YES → Proceed to Phase 1 (benchmark)
└── NO → Two paths:
    ├── Gate −1 fails → Annotation approach needs rework
    ├── Gate 0 fails → Target strategy needs rework
    ├── Gate 1 fails → Identity preservation needs new method
    └── Gate 2 fails → Content preservation needs new method
```

---

## 8. LEGACY ENGINEERING — WHAT WAS BUILT, WHAT FAILED

### Seed-VC (Rejected)
- Open-source voice conversion model
- Rejected because: changes speaker identity, doesn't preserve content exactly
- Lesson: existing S2S models don't meet the linguistic contract

### FACodec Branch
- Factorized audio codec from Plachta et al.
- Successfully integrated: can encode/decode with factorized latents
- **Critical lesson:** timbre path in FAcodec is entangled with content — you can't cleanly isolate accent without affecting timbre
- FAcodec is a codec, not an accent converter — it compresses, it doesn't transform

### Stage A (Invalidated)
- Initial accent conversion attempt
- **Why invalidated:** Used different speakers for source and target (not same-speaker)
- Speaker preservation claims from Stage A are **not valid evidence**

### Architecture Bake-off (Phase 2)
- 5 candidates evaluated on paper: A (Streaming AC), B (Articulatory DDSP), C (Token Translation), D (Minimal Hybrid), Sparse Repair
- **Selection: Candidate D** — lowest latency, simplest architecture, best streaming characteristics
- **Caveat:** Selection was paper-based, not empirical. No real training or listening test was done.

### What Survives and Can Be Reused
- Streaming session abstractions
- Causality harness
- Latency profiler and state-growth tests
- Chunk/lookahead sweep tooling
- Experiment/manifest logic
- Some simple baseline/Candidate D code (if clean)

---

## 9. MODEL LANDSCAPE

### Existing Real-Time Audio-to-Audio Models Evaluated
| Model | License | Why Rejected for v1 |
|-------|---------|---------------------|
| **RVC** | MIT | Changes timbre along with accent |
| **Beatrice** | Unknown | Too experimental |
| **CosyAccent** | Apache 2.0 | Different project, different authors (kept separate) |
| **Seed-VC** | MIT | Identity not preserved |
| **ppg2ppg** | Unknown | Not suited for accent conversion |
| **seq2seq-vc** | Unknown | High latency |

### The Decision Tree
```
Can existing real-time engine learn accent conversion?
├── YES → Benchmark generalization → Stream it → Optimize/export → MVP
└── NO → Can a simple baseline learn it offline?
    ├── NO → Revisit representation/target/loss
    └── YES → Failure-driven architecture:
        ├── Context bottleneck → streaming Conformer/Emformer
        ├── Synthesis bottleneck → lighter vocoder/DDSP
        ├── Pronunciation mapping → token/phonetic representation
        └── Whole-speech identity failure but sparse repair works → sparse-repair product branch
```

### Proprietary Model Architecture (Phase 3+)
Only pursued if existing engines prove unsuitable. The document describes a custom architecture:
- **Encoder:** Causal, processes mel frames with lookahead
- **Accent bottleneck:** Compressed representation of pronunciation patterns
- **Mapper:** Transforms accent representation while preserving identity
- **Synthesizer:** Lightweight vocoder (HiFi-GAN or DDSP-inspired)
- **Speaker encoder:** Separate pathway for identity conditioning

### Key Design Principles
1. **Causality first** — no future frames in production
2. **Speaker disentanglement** — identity must flow through a separate pathway
3. **Phonetic supervision** — phone-level targets for pronunciation
4. **Factorized latents** — accent/modification separate from timbre/prosody

---

## 10. STREAMING & RUNTIME ARCHITECTURE

### Audio Pipeline (Physical Mic → Virtual Microphone)
```
Physical mic → DC removal → Resample → Noise suppression → AEC → Gain normalization → VAD → Accent model → Post-DSP/overlap → Virtual microphone
```

### Components Discussed
- **VAD:** Silero VAD (MIT license, already used in legacy workflows)
- **Noise suppression:** RNNoise (BSD-3-Clause, don't build custom)
- **AEC:** Acoustic echo cancellation where topology requires it
- **Keep Phase 0 clean:** Don't add noise/AEC during target generation

### Real-Time Metrics Definitions
| Metric | Definition |
|--------|-----------|
| **RTF** | Real-Time Factor: processing time / audio duration |
| **Latency** | Time from last sample entering to first sample exiting |
| **Lookahead** | How many future frames the model needs |
| **Algorithmic latency** | Theoretical minimum latency from model architecture |

### Performance Targets (Phase 5+)
- RTF < 0.1 (process 1s audio in < 100ms)
- End-to-end latency < 200ms
- Model fits in 2GB VRAM

---

## 11. CROSS-PLATFORM ARCHITECTURE

### Platform Priority
1. **Windows first** — BPO agents use Windows desktops
2. **macOS** — for development, testing, and executive/manager use
3. **Linux** — server-side processing, CI/CD

### NOT Windows-Only
The document explicitly says Windows is the first target but the architecture must be cross-platform. Reasons:
- Management/QA may use macOS
- Server-side batch processing may run on Linux
- Future expansion beyond BPO

### Stack Discussion
- **Training:** Python/PyTorch on Linux
- **Inference:** C++/ONNX runtime on Windows
- **Desktop wrapper:** Electron or Tauri for admin/config UI
- **gRPC:** Service communication between components

---

## 14. FACodec INTEGRATION

### What FACodec Provides
- Factorized quantized representation of speech
- 8 codebooks total:
  - 2 content codebooks (z_c1, z_c2)
  - 1 prosody codebook (z_p)
  - 3 residual detail codebooks (z_r)
  - 2 timbre codebooks (z_t)

### Critical Lesson
The timbre path in FAcodec is **entangled** with content. You cannot cleanly extract "accent-only" latents. The codec compresses speech, it doesn't separate accent from identity.

### Current Status
- FAcodec encode/decode works
- Factorized latents can be extracted
- The isolation of accent-bearing components from timbre-bearing components is **not yet solved**

### Frame Rate Issue (from analysis)
- Code reports 80 fps
- Paper reports 50 fps
- **Risk:** If wrong, all training data is misaligned
- **Action needed:** Verify against FAcodec source code

---

## 15. TRAINING PLAN

### Datasets
- **Primary:** CMU Arctic (public, diverse speakers)
- **Secondary:** IndicTTS, CSS10 (Indian English speakers)
- **Synthetic:** Generated targets from Phase 0

### Preprocessing
- Resample to 24kHz (FAcodec native rate)
- Mono downmix
- Normalize amplitude
- Extract FAcodec latents (cached)

### Training Schedule
| Stage | Epochs | Batch Size | What It Trains |
|-------|--------|------------|----------------|
| Stage A | 50 | 32 | Encoder + accent mapper only |
| Stage B | 100 | 16 | Add synthesizer |
| Stage C | 50 | 8 | Fine-tune all, phone-level loss |

### Loss Functions
- **Reconstruction:** Mel spectrogram L1
- **Identity:** Speaker embedding cosine similarity
- **Content:** Phone-level cross-entropy
- **Accent:** Phonetic distance between predicted and target phones
- **Adversarial:** Discriminator for naturalness

### Key Constraint
Training requires FAcodec frozen (not fine-tuned). The codec is the fixed representation; only the accent transformation layers are trained.

---

## 16. EVALUATION & BENCHMARKS

### Metrics by Dimension
| Dimension | Metric | Method |
|-----------|--------|--------|
| **Content** | WER | Whisper ASR transcription |
| **Content** | CER | Character error rate (phone-level) |
| **Identity** | Speaker similarity | ECAPA-TDNN embedding cosine |
| **Acoustic** | STOI | Short-Time Objective Intelligibility |
| **Acoustic** | PESQ | Perceptual Evaluation of Speech Quality |
| **Acoustic** | MCD | Mel Cepstral Distortion |
| **Naturalness** | MOS | Mean Opinion Score (human listening) |

### Production Gates
| Gate | Threshold | Meaning |
|------|-----------|---------|
| WER | < 10% | Content preserved |
| Speaker sim | > 0.85 | Identity preserved |
| STOI | > 0.85 | Intelligible |
| PESQ | > 2.5 | Natural-sounding |
| MOS | > 3.5 | Human-acceptable |

### Benchmark Design Principles
- Speaker-disjoint: no speaker appears in both train and test
- Leakage-resistant: SHA-256 audio integrity, hash verification
- No data leakage between phases

---

## 17. LICENSING, PROVENANCE & PRIVACY

### Open-Source Components
| Component | License | Risk |
|-----------|---------|------|
| PyTorch | BSD-3 | Low |
| FAcodec | Check required | Medium |
| Whisper | MIT | Low |
| Silero VAD | MIT | Low |
| RNNoise | BSD-3 | Low |
| SpeechBrain | MIT | Low |
| Librosa | ISC | Low |

### Privacy Architecture
- **Local-first:** No raw audio upload by default
- **Analytics opt-in:** Usage metrics can be sent, not required
- **Model distribution:** Signed model packages, not streaming weights
- **Fleet management:** On-prem, no external dependency

### Data Provenance
- All training data must have clear provenance
- No scraped or unclear-license data in v1
- CMU Arctic and similar permissive licenses only

---

## 18. BUSINESS ECONOMICS

### Revenue Model
- Per-seat licensing (not per-minute)
- Illustrative: $40/seat/month
- 1,000 seats = $40K MRR
- No cloud processing cost after deployment

### Competitive Landscape
| Competitor | Approach | Differentiation |
|------------|----------|----------------|
| Sanas | Real-time cloud | We're on-device, no upload |
| Microsoft | Cloud TTS | We preserve identity, not synthetic |
| Google | Cloud TTS | Same |
| Custom BPO solutions | Rule-based DSP | We're ML-based, more natural |

### Go-to-Market
- Direct BPO sales (not consumer)
- Pilot program: 10-50 seats, 6 weeks
- Case study from pilot → broader sales
- Integration with existing BPO phone systems

---

## 19. MASTER RISK REGISTER

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Phase 0 fails (target not constructible) | Medium | Project stops | Failure is informative; pivot to simpler target |
| Identity not preserved at scale | Medium | Core thesis fails | Use stronger speaker encoder, more training data |
| FAcodec timbre entanglement unsolvable | Low | Need new codec | Alternative: use different factorization method |
| License issue with FAcodec | Low | Can't use codec | Replace with open-source alternative (DAC, EnCodec) |
| BPO buyers won't pay | Medium | No revenue | Prove ROI in pilot phase |
| Real-time latency too high | Medium | Can't deploy | Use smaller model, quantization, pruning |
| Accent too varied to model | Low | Limited market | Focus on "standard" Indian English only |

### Unresolved Questions
1. Can human listeners reliably distinguish identity-preserved targets?
2. Does the linguistic contract cover all relevant accent dimensions?
3. What is the minimum dataset size for acceptable quality?
4. Can the model generalize to unseen speakers?
5. What is the acceptable failure rate for accent conversion?

---

## 20. IMMEDIATE EXECUTION PLAN

### Next Actions (Ranked by Priority)
1. **Execute TGFP v2 Step 0** — One speaker, one sentence, Strategy B, hand-built target
2. **Fix broken imports** — 8 broken imports, 4 missing deps (5 minutes each)
3. **Verify FAcodec frame rate** — 80 vs 50 fps discrepancy
4. **Build training driver** — Phase 1 has no train.py
5. **Write tests for benchmark/ and models/** — 0% coverage on 76 files
6. **Set up test dataset** — CMU Arctic, 2-3 Indian English speakers
7. **Run Gate −1** — Annotation validity test

### Decision Tree
```
TGFP v2 Step 0 complete?
├── YES → Can a human hear accent change?
│   ├── YES → Can they identify the same speaker?
│   │   ├── YES → Gate 0 passes → proceed to Gate 1
│   │   └── NO → Identity not preserved → revise target strategy
│   └── NO → Gate 0 fails → target not usable
└── NO → Is the target strategy wrong?
    ├── YES → Try Strategy A or C
    └── NO → Technique problem → fix alignment/editing
```

---

## 21. APPENDIX D — LINK REGISTER

The dossier contains 77 links across these categories:
- **Academic papers:** FAC-FACodec (arXiv:2510.10785v2), FAcodec, Seed-VC
- **GitHub repos:** snakers4/silero-vad, xiph/rnnoise, Plachta/FAcodec
- **Datasets:** CMU Arctic, CSS10, IndicTTS, VCTK
- **Tools:** Whisper, SpeechBrain, Librosa, torchaudio
- **Licenses:** Various (MIT, BSD-3-Clause, Apache 2.0, ISC)

---

## 22. CRITICAL GAPS BETWEEN DOSSIER AND CODEBASE

| Dossier Says | Code Has | Gap |
|--------------|----------|-----|
| Phase 2 complete | Candidates A/B/C/D + Sparse Repair exist but are STUBS | Selection was paper-based, not empirical |
| Candidate D selected | MinimalHybridCandidate exists with `TODO` implementations | No real training or evaluation |
| "Speaker preservation validated" | Speaker embedding code exists, no validation data | Never run at scale |
| "Content preservation validated" | WER/CER metrics implemented | Only on Whisper, not word-level |
| "RTF proves AccentEdge is fast" | Latency profiler exists, no production benchmarks | Theoretical only |
| Training driver needed | `trainer.py` exists but no `train.py` entry point | Can't actually train |
| Frame rate 50 fps | Codec code uses 80 fps | Could invalidate all training data |
| FAcodec timbre entangled | FactorizedLatents dataclass exists | No solution implemented yet |

---

## 23. FINAL SYNTHESIS

The dossier ends with: *"The Architecture to Commit To Without Overbuilding"* — but the content is just the closing line. The synthesis is in the Decision Tree (§20.2):

1. **Start with existing real-time engines** (RVC, Beatrice) — can they learn accent conversion?
2. **If yes:** benchmark → stream → optimize → MVP
3. **If no:** try simple baseline offline
4. **If simple baseline fails:** revisit the representation/target/loss
5. **If simple baseline succeeds:** failure-driven architecture — let the failure mode dictate the next architecture choice

The core philosophy: **don't overbuild. Let evidence dictate architecture.**

---

## 24. KEY INSIGHTS

1. **This is a research project, not a product.** The dossier is explicit: no claim is proven until Phase 0 passes all 4 gates.

2. **The linguistic contract is the north star.** Every engineering decision must preserve: same speaker, same words, same emotion. Any change to these three is a failure.

3. **On-device is the competitive moat.** The cloud idea was killed by economics and privacy. Local-first is both cheaper and more defensible.

4. **Failure is informative.** If Phase 0 fails, the project doesn't "fail" — it learns something. The dossier frames failure as data, not disaster.

5. **Existing models are rejected for good reasons.** RVC, Seed-VC, and others change speaker identity. The thesis requires a fundamentally different approach.

6. **The codebase is at "Phase 0 execution" stage.** Phase 2 is labeled "complete" but was paper-based selection only. No model has been trained, no listening test done, no benchmark validated at scale.

7. **Frame rate mismatch is the most urgent technical risk.** If the codec runs at 80fps but should be 50fps, all cached latents are wrong.

8. **License review is non-negotiable.** FAcodec's license must be verified before training. If incompatible, the entire codec path needs replacement.

---

*Analysis complete. Full dossier content preserved in `_analysis/dossier_sections.json` and `_analysis/dossier_tables.md`.*
