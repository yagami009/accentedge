
## TABLE 1 (8 rows)

| Label | Meaning | How to use it |
| ACTIVE / SOURCE OF TRUTH | Current product/scientific constraint that wins ov | Build and report against this unless a dated decis |
| CURRENT DECISION | A design/business decision reached in the current  | Treat as the preferred direction, subject to the s |
| POST-GATE PROPOSAL | A detailed production choice that becomes relevant | Do not build prematurely; retain for later phases. |
| EXPLORATORY | Candidate, benchmark, or hypothesis discussed for  | Measure before adopting. |
| LEGACY EVIDENCE | Real work/results from an older branch that remain | May inform debugging, but does not prove current m |
| INVALIDATED | A prior claim/metric/experiment that was found not | Keep for lessons only; never cite as proof. |
| LICENSE CHECK REQUIRED | Code/weights/data may have different licenses or u | Legal/provenance review before training, target ge |


## TABLE 4 (13 rows)

| Dimension | Current position |
| User | India-based BPO/contact-centre agent serving US cu |
| Buyer | BPO, contact centre, outsourced support operation, |
| Direction | Agent → customer first. |
| Source | Indian English. |
| Target | Neutral/General US pronunciation profile. |
| Inference | Direct speech-to-speech; no ASR→LLM→TTS in product |
| Identity | Same speaker must remain perceptually recognizable |
| Content | Exact words and critical entities must remain inta |
| Emotion | Preserve emotional intent and conversational flow. |
| Privacy | Local-first; no raw call-audio upload by default. |
| Endpoint | Windows first commercial MVP; architecture should  |
| Softphone integration | Virtual microphone / OS audio device rather than d |


## TABLE 6 (4 rows)

| Example | Calculation | Implication |
| Light usage | $1 × 60 min/day × 22 days = $1,320/agent/month | Far above a $60/user/month public benchmark. |
| 4 hours/day | $1 × 240 min/day × 22 days = $5,280/agent/month | Implausibly expensive for broad BPO rollout. |
| 100 seats at 4h/day | $528,000/month | Versus $6,000/month equivalent for 100 users in th |


## TABLE 8 (7 rows)

| Packaging concept | Illustrative discussion range | Status |
| Pilot | $20–40 / agent / month | Exploratory pricing, not market-validated. |
| Standard | $40–70 / agent / month | Exploratory. |
| Enterprise | $60–100+ / agent / month | Exploratory; could bundle controls/support. |
| Private/on-prem | Annual enterprise contract | Preferred for privacy-sensitive deployments. |
| Custom accent/model | Additional enterprise fee | Potential services/IP line. |
| API | Per audio minute | Optional secondary product, not primary BPO motion |


## TABLE 11 (13 rows)

| Dimension | V1 treatment / note |
| Rhoticity | Transform where source realization differs; many e |
| Vowel quality | Selected TRAP / LOT / THOUGHT / GOAT / FACE-like s |
| Vowel reduction | Transform where appropriate; not purely local beca |
| /θ/ and /ð/ | Transform dental-fricative realization when needed |
| Voiceless stop aspiration | Target stressed-syllable onset realization. |
| Retroflex/alveolar /t d/ | Move toward target realization where deviant. |
| /v/–/w/ | Correct only where needed; many speakers already p |
| Intervocalic /t/ flapping | Target behavior where appropriate. |
| Cluster epenthesis | Reduce target-inconsistent epenthetic vowels where |
| Lexical stress | Correct within bounded scope. |
| Phone duration | May change; do not impose a fake local constancy. |
| Word/phrase timing | Must be calibrated from natural same-speaker cross |


## TABLE 12 (2 rows)

| Must transform | Must preserve | May change / calibrated |
| Selected segmental pronunciation; bounded lexical  | Exact lexical content; speaker identity; emotional | Phone duration; limited word timing; some prosodic |


## TABLE 14 (6 rows)

| Gate | Question | Kill / redirect condition |
| −1 | Do we know where relevant phones are and which sou | Manual annotation/alignment is not tractable even  |
| 0 | One speaker, one sentence, Strategy B: does it sti | Identity collapses immediately. |
| 1 | Can teacher strategies repeatedly generate targets | No strategy clears the admission gate. |
| 2 | What does genuine same-speaker cross-accent change | Gold/reference data shows the v1 contract itself i |
| 3 | Benchmark + streaming student. | Reached only after Gates 0–2 pass. |


## TABLE 15 (4 rows)

| Condition | Construction | Purpose |
| NB | Resample to 8 kHz; G.711 μ-law round trip. | Does benefit survive telephony bandwidth/codec? |
| NOISY | Call-centre babble around 15 dB SNR + headset-resp | Can target generation tolerate realistic input? |
| NB+NOISY | Both. | Proxy for actual production condition. |


## TABLE 16 (5 rows)

| Strategy | Flow | Primary failure mode |
| A — source-conditioned native TTS | transcript + source F0 + source durations + speake | Speaker embedding may carry accent back into the o |
| B — native realization first, identity second | transcript → native-US TTS → identity/timbre conve | Identity/VC stage can re-introduce accent from ref |
| C — whole-utterance analysis/resynthesis with spar | analyze whole utterance → edit selected latent/con | Local discontinuities, token-decision threshold, l |
| GOLD — natural bidialectal recording | same human naturally renders matched sentence in I | Not a production method; empirical calibration ref |


## TABLE 17 (7 rows)

| Condition | Speaker | Accent/style/session | What it establishes |
| A | same | Indian baseline | Ceiling / measurement noise. |
| B | same | Indian, different session | Session nuisance floor. |
| C | same | Indian, changed rate/emotion | Style nuisance floor. |
| D | same | code-switched US, matched sentence | Distribution that matters for identity/timing. |
| E | different | Indian | Imposter, accent-matched. |
| F | different | US | Different speaker + accent. |


## TABLE 18 (9 rows)

| Code | Dimension |
| RHO | Rhoticity |
| FLAP | Intervocalic /t/ |
| TH | Dental fricatives |
| ASP | Onset aspiration |
| RET | Retroflex → alveolar |
| VW | /v/–/w/ contrast |
| RED | Vowel reduction |
| STR | Lexical stress |


## TABLE 19 (5 rows)

| Level | Constraint philosophy |
| Phone | Unconstrained where pronunciation requires local d |
| Word | Derive acceptable bands from natural same-speaker  |
| Phrase | Earlier provisional teacher/product values (<5% /  |
| Turn | Measure cumulative drift over spontaneous ~30s seg |


## TABLE 20 (7 rows)

| Metric | Pilot/product hypothesis | Teacher target hypothesis |
| Naturalness MOS | ≥4.0 [P] | ≥4.3 [P] |
| Human speaker similarity | ≥4.0 [P] | ≥4.3 [P] |
| Critical entity accuracy | ≥99% | 100% |
| Identity | — | ≤ d_gold + CI |
| Phrase-duration drift | <2% [P] | <5% [P] |
| Damage rate | — | ≤2% [P] |


## TABLE 21 (5 rows)

| Outcome | Meaning | Next move |
| FULL-S2S PASS | Whole-speech teacher clears admission on most utte | Proceed toward full benchmark and direct causal S2 |
| SPARSE-REPAIR PASS | Whole-speech identity fails but sparse strategy wo | Redefine around real-time pronunciation repair; pr |
| TEACHER FAIL / GOLD PASS | Natural human cross-accent references pass; synthe | Thesis survives; supervision/identity transfer is  |
| FUNDAMENTAL FAIL | Even controlled natural cross-accent change cannot | Stop downstream ML; re-scope to lighter intelligib |


## TABLE 22 (6 rows)

| Area | Pilot measurement |
| Scale | Start 3–5 agents; expand to 10–20 if stable. |
| Technical | P50/P95 latency, CPU, RAM, session duration, crash |
| Agent | Still sound like you? voluntarily keep enabled? le |
| Business | Clarification/repeat requests, AHT, FCR, QA, CSAT, |
| Continuation | Agents can use it; calls stay stable; listeners pe |


## TABLE 23 (9 rows)

| Finding | Legacy evidence | Current interpretation |
| Successful conversions | 5 successful full-utterance WAV conversions among  | Real integration/debugging achievement; not accent |
| Model load | ~9.19 s CPU; roughly 1.2–1.4 GB peak RAM depending | Useful operational fact only. |
| Inference | ~37–47 s wall-clock for ~12 s audio on CPU. | True inference-only RTF ≈3.1–3.9; not real-time. |
| Reported RTF | 0.003–0.004 in legacy metrics. | INVALIDATED as product evidence because timing cal |
| Device | MPS appeared available but all recorded runs used  | Acceleration path unverified. |
| Architecture | Semantic encoder + CAMPPlus style + RMVPE F0 + CFM | Full-utterance zero-shot VC path. |
| Streaming | WebSocket/StreamingPipeline scaffolding existed; c | NOT PROVEN. |
| Product thesis | Seed-VC is a voice converter; output could copy ta | Core risk; generic VC ≠ same-speaker accent normal |


## TABLE 24 (9 rows)

| ID | Issue | Lesson for current design |
| P0 | Tensor .numpy() without detach caused runtime fail | Inference wrappers need deterministic no-grad/runt |
| P1 | Misleading RTF timing. | Keep load, compute, algorithmic, first-output, end |
| P1 | Unbounded queues in capture/playback. | All production buffers bounded; degrade/bypass rat |
| P2 | Status flags dropped. | Surface overruns/underruns as metrics/health. |
| P2 | No input-device selection. | Endpoint UX must explicitly bind headset/mic. |
| P2 | Model load concurrency/idempotency risk. | Thread-safe lifecycle and one model instance per i |
| P0 risk | Voice-conversion architecture mismatch. | Validate same-speaker accent behavior before optim |
| P2 risk | Unpinned dependencies/checkpoints. | Lockfile, hashes, model manifest, SBOM/provenance. |


## TABLE 25 (7 rows)

| Track | What was changed | Current evidence status |
| Reconstruction adapter | Corrected `forward_v2` unpacking to [z_p, z_c, z_r | Code-level fix; real reconstruction gate still mus |
| Frame rate | Corrected phoneme/frame assumptions from ~50 fps t | Unit/math corrected; must validate on real alignme |
| Phoneme conditioning | text→phones via phonemizer/eSpeak; audio alignment | Real implementation; many tests used mocked extern |
| Tests | 34 passed, 2 skipped, 1 warning in the recorded st | Software correctness only; not scientific validati |
| Real-data gaps | Real z_c extraction/shape, real end-to-end phone a | UNVERIFIED. |
| Architectural gap | Questions remained around z_c2 recomputation and e | Must resolve only if this branch is revived after  |


## TABLE 27 (14 rows)

| Model / engine | Realtime today? | Training / FT | License posture discussed | Role for AccentEdge |
| CosyAccent | Whole-WAV inference; not a purpose-built live stre | Public repo currently inference-heavy; checkpoint  | HF model card MIT; audit dependencies/teacher data | Highest-priority accent-specific teacher/reference |
| TokAN-Legacy | VAD chunking for long files; not identical to low- | Full training and fine-tuning recipe. | MIT repo; training uses datasets with their own li | Best open accent-normalization training reference. |
| FACodec / NaturalSpeech3 codec | Codec component, not a finished accent product. | Training/research ecosystem available. | Current HF checkpoint metadata Apache-2.0; verify  | Factorized foundation for proprietary model if nee |
| seq2seq-vc | Not selected for production live path. | Full toolkit; L2-ARCTIC FAC recipe. | MIT code; dataset recipe includes noncommercial L2 | Reproducible FAC research baseline. |
| PPG2PPG | Research system. | Training pipeline. | Apache-2.0 code as discussed; verify exact assets. | Accent-specific research baseline. |
| CosyVoice | VC function exists; heavier generation stack. | Training/SFT available. | Apache-2.0 repo ecosystem; exact checkpoint audit  | High-quality resynthesis/VC experiment. |
| SpeechT5 VC | Not ideal for strict live-call latency. | Fine-tunable. | MIT model card. | Commercially simpler audio-to-audio baseline. |
| RVC official | Yes; mature real-time ecosystem. | Very easy custom voice training. | Official project MIT; forks/assets differ. | Fast MVP/student experiment; normally timbre VC, n |
| OpenVoice V2 | Fast voice/tone conversion, but not primary stream | Training less first-class in public workflow. | Official project states MIT/free commercial use; e | Optional timbre/identity stage. |
| Seed-VC | Has real-time support upstream; legacy AccentEdge  | Fine-tuning/zero-shot capabilities. | GPL-3.0; problematic for closed-source proprietary | Strong technical reference/lab benchmark, not pref |
| Vevo / Vevo-Style | Research conversion system. | Architecture useful. | Released Amphion weights CC BY-NC 4.0. | Study architecture; do not ship/use NC weights for |
| Beatrice VST / V2 ecosystem | Yes; explicitly real-time VC DSP engine. | Training ecosystem exists. | Core engine MIT; bundled JVS corpus editions/non-c | Very important low-latency MVP/student candidate. |
| w-okada voice-changer | Realtime wrapper/server; supports multiple VC fami | Depends on selected backend. | Wrapper contains MIT licenses plus license notices | Excellent laboratory/prototyping harness; not blin |


## TABLE 30 (6 rows)

| Experiment | Candidate | Question |
| E1 | RVC | Can the easiest realtime engine learn same-speaker |
| E2 | Beatrice | Can a low-latency engine preserve identity better  |
| E3 | Seed-VC realtime upstream | What quality/latency/zero-shot behavior is achieva |
| E4 | Validated teacher → RVC/Beatrice | Can synthetic paired supervision compress accent q |
| E5 | Selected model → ONNX/native | Can the winning student be exported/optimized for  |


## TABLE 32 (9 rows)

| Item | Proposed value / rule | Status |
| Internal speech sample rate | 16 kHz mono for the ML path. | Proposed; aligns with many speech models and telep |
| Capture device rate | May be 44.1/48 kHz; resample immediately to model  | Proposed. |
| Internal numeric format | float32 in DSP/model pipeline; PCM16 for network t | Proposed. |
| Audio frame | 20 ms = 320 samples @16 kHz. | Proposed standard framing. |
| Model context | ~160–320 ms overlapping context. | Exploratory; final from chunk/lookahead sweep. |
| Crossfade/overlap-add | Smooth boundaries between processed chunks. | Required if chunked waveform synthesis creates sea |
| Queues | Bounded only. | ACTIVE runtime safety rule. |
| Failure | Crossfade to original mic rather than silence. | ACTIVE fail-open rule. |


## TABLE 33 (7 rows)

| Metric | Definition / why it matters |
| RTF | processing_seconds / audio_seconds. RTF <1 is nece |
| Algorithmic latency | Future audio/lookahead structurally required befor |
| Compute latency | Wall-clock time spent processing a frame/chunk. |
| First-output latency | Time from first accepted input to first transforme |
| End-to-end added latency | Frame accumulation + lookahead + compute + synthes |
| Backlog | Unprocessed audio accumulation. Must remain bounde |


## TABLE 34 (5 rows)

| RTF | Interpretation |
| 2.0 | Impossible for sustained live stream without backl |
| 1.0 | Barely keeps up; no concurrency/headroom. |
| 0.25 | ~4× real-time raw throughput before overhead; earl |
| 0.05 | ~20× raw real-time throughput before overhead; exc |


## TABLE 36 (7 rows)

| Layer | Windows | macOS | Linux |
| Audio capture | WASAPI | CoreAudio | PipeWire |
| Virtual microphone | SysVAD/WaveRT-derived virtual device or appropriat | Audio Server Driver Plug-in / HAL virtual device | PipeWire virtual source/loopback |
| NVIDIA acceleration | TensorRT/CUDA | N/A | TensorRT/CUDA |
| Apple acceleration | N/A | CoreML → CPU/GPU/Neural Engine | N/A |
| Intel/other | WinML/OpenVINO/CPU depending target | CPU/CoreML | OpenVINO/CPU; MIGraphX for AMD where supported |
| UI | Shared Flutter proposed; native WinUI remains an o | Shared Flutter; SwiftUI optional native | Shared Flutter; GTK optional native |


## TABLE 37 (5 rows)

| Mode | Audio path | Pros | Cons |
| Cloud/SaaS GPU | BPO → Internet/VPN → AccentEdge GPU → returned aud | Central updates and easiest centralized ops. | Latency, privacy, egress, GPU-per-minute economics |
| On-prem software | PBX/agent LAN → local AccentEdge GPU gateway → sof | Audio stays inside customer network; easier pilot  | Customer appliance/server ops; concurrency sizing  |
| On-device endpoint | Physical mic → local model → virtual mic | Best privacy/unit economics; minimal central GPU d | Requires model distillation/optimization across he |
| AccentEdge appliance | Rackable validated hardware + software | Simple enterprise procurement/support boundary. | Hardware logistics/support; risk of premature cape |


## TABLE 38 (7 rows)

| Environment | GPU | CPU | RAM | Storage | Purpose |
| Developer | RTX 4070/4080/4090 16–24GB | 12–16 cores | 64GB | 2TB NVMe | Research/benchmark. |
| On-prem pilot | 1× NVIDIA L4 24GB | 16–24 cores | 64–128GB ECC | 2TB NVMe | First live concurrency benchmark. |
| Small production | 2× L4 24GB | 24–32 cores | 128GB ECC | 2–4TB | Multiple streams + some failover/headroom. |
| Performance production | 1–2× L40S 48GB | 32–48 cores | 128–256GB ECC | 4TB | Higher throughput / heavier model. |
| Larger enterprise | 2–4× L40S | 48–64 cores | 256GB ECC | 4–8TB | Only after measured demand. |
| Training/R&D | 2–4× 48–96GB class GPU | 32–64 cores | 256–512GB | 4–12TB | Fine-tune/full training/teacher experiments. |


## TABLE 39 (4 rows)

| GPU | Verified vendor specs relevant here | AccentEdge interpretation |
| NVIDIA L4 | 24GB memory, ~300GB/s, 72W, low-profile single-slo | Attractive pilot inference GPU due power/form fact |
| NVIDIA L40S | 48GB GDDR6 ECC, 864GB/s, up to 350W. | Heavier/high-throughput inference or R&D if L4 is  |
| RTX PRO 6000 Blackwell Server Edition | 96GB GDDR7 ECC, ~1597GB/s, up to 600W, very high t | Overkill for initial inference; useful combined R& |


## TABLE 42 (23 rows)

| Layer | Preferred stack discussed | When |
| Research/modeling | Python, PyTorch, torchaudio, NumPy, SciPy, SoundFi | Now for target/model experiments. |
| Experiment config | Hydra | When experiments become numerous enough to justify |
| Experiment tracking | MLflow | Post-gate, when real runs need lineage. |
| Dataset/versioning | DVC + explicit manifests/checksums | As soon as production-safe datasets are ingested. |
| Model interchange | ONNX | After a model is good enough to optimize. |
| Native inference | ONNX Runtime C++ | Phase 4 / endpoint optimization. |
| NVIDIA | TensorRT/CUDA EP | On-prem/Windows/Linux NVIDIA targets. |
| Apple | CoreML EP / native CoreML path | macOS/Apple Silicon port. |
| Intel | OpenVINO EP where beneficial | Windows/Linux Intel CPU/iGPU/NPU. |
| Core runtime | C++20 | Streaming/audio/inference once scientific value is |
| Desktop UI | Flutter desktop + FFI | Cross-platform packaging phase. |
| Windows audio | WASAPI + virtual-audio device | Windows endpoint phase. |
| macOS audio | CoreAudio + Audio Server Driver Plug-in | macOS endpoint. |
| Linux audio | PipeWire | Linux endpoint/thin client/server. |
| On-prem audio transport | gRPC bidirectional streaming + mTLS | Only if using LAN GPU gateway. |
| Control-plane API | FastAPI + Pydantic + SQLAlchemy + Alembic | Post-MVP fleet/licensing phase. |
| Database | PostgreSQL | Control plane. |
| Cache/ephemeral | Redis | License/token cache, rate limit, ephemeral device  |
| Dashboard | Next.js + TypeScript + React + Tailwind | Admin/fleet phase, not Phase 0. |
| Object/model storage | Azure Blob / S3-compatible | Model registry and signed artifacts. |
| Observability | OpenTelemetry + Prometheus + Grafana + Loki | Runtime/pilot operations. |
| Packaging/deploy | Docker; Terraform for infra | Server/control-plane production. |


## TABLE 44 (9 rows)

| Dataset | What it gives | Scale discussed | License posture | Recommended use |
| Common Voice Indian Accent (HF derivative) | Indian-accent English source speech. | 163.89h; 110,088 recordings; ~4.12GB. | HF card CC0, derived from Common Voice v21; preser | First large source-accent corpus; speaker diversit |
| Kaggle Indian English Accent Audio | Regional Indian-English folders. | 6.01GB; 8,115 files. | Kaggle page CC0; provenance/ownership of uploaded  | Experimental source diversity; production only aft |
| CMU ARCTIC | Phonetically designed native US English speakers. | ~1,100–1,200 utterances per speaker; multiple US s | Permissive CMU/FestVox terms discussed; verify exa | Native-US phonetics/reference/evaluation/teacher c |
| Google FLEURS en_us | US English speech. | Subset of multilingual FLEURS. | CC BY 4.0 metadata. | US-accent distribution/classifier/reference. |
| LibriTTS-R | Large high-quality multi-speaker English. | Hundreds of hours; discussion referenced ~755k row | Commonly CC BY 4.0; verify original OpenSLR corpus | General speech encoder/decoder/pretraining. |
| VCTK | Multi-speaker English with accent diversity. | ~44h; ~110 speakers. | CC BY 4.0 metadata. | Teach speaker≠accent disentanglement; evaluation. |
| AI4Bharat IndicVoices | Large Indian-language speech. | Very large multi-language corpus. | CC BY 4.0 HF card; gated contact sharing. | Indian acoustic/speaker representation pretraining |
| Project Vaani | Large geographically diverse Indian speech. | Discussion referenced ~31k hours/current expanding | CC BY 4.0 on discussed release; verify per version | Optional Indian acoustic robustness/pretraining; d |


## TABLE 45 (6 rows)

| Source | Relevant material | Commercial posture | Use |
| LDC-IL / CIIL | Indian English Bengali/Kannada variants; raw and s | Explicit commercial user path exists, but terms fo | High-quality Indian-English phonetics and regional |
| Defined.ai | Indian-English simulated banking/insurance/telco/r | Commercial vendor; negotiate exact ML training/der | Strong domain adaptation after feasibility. |
| InfoBayAI | India/US call-center previews on Hugging Face disc | Public preview ≠ full commercial rights; license f | Matched domain on India and US sides. |
| Axon / Kaggle preview | English call-center speech; discussion referenced  | Sample/noncommercial/public terms may differ from  | Domain ASR/acoustics/call-center realism. |
| ELRA/ELDA catalog | Accented English / Indian-English mobile speech re | Commercial catalog; exact item/rights must be re-v | Curated accent/domain data if rights justify cost. |


## TABLE 46 (5 rows)

| Dataset | Why useful | Restriction / decision |
| L2-ARCTIC | 24 non-native speakers across Arabic, Mandarin/Chi | CC BY-NC 4.0 in discussed HF release. Research/eva |
| Speech Accent Archive (GMU) | Large cross-accent comparison where speakers read  | Current site states CC BY-NC-SA 4.0. Keep out of c |
| Whissle Indian-English assets | Indian-accent-optimized STT / tech-interview asset | Inference-only license explicitly prohibits traini |
| Random HF/Kaggle YouTube-derived sets | May add speaker/accent variety. | A dataset card claiming CC-BY/CC0 cannot magically |


## TABLE 48 (4 rows)

| Stage | Illustrative scale | Audio math | Purpose |
| Initial proprietary scale-up | 30–50 speakers × 20–30 min per condition | Example 50 × 30 min =25h source +25h target =50h t | Enough diversity to train/calibrate a first seriou |
| Larger corpus | ~200 speakers ×45 min per condition | ~75h source +75h target =150h total. | Regional/L1 diversity and better generalization. |
| Domain expansion | Add BPO scripts, spontaneous turns, noise/telephon | Depends on validated collection protocol. | Call-center robustness and business vocabulary. |


## TABLE 49 (12 rows)

| Metric/goal | Value discussed | Status in master |
| Pilot P95 added latency | ≤250–300 ms | Current source-of-truth provisional engineering ob |
| Stronger commercial P95 | ≤200 ms | Current later objective. |
| Earlier “hard ceiling” | ~250 ms | Exploratory architecture target; subordinate to me |
| RTF pilot | ≤0.6 | Current source-of-truth provisional. |
| Earlier launch RTF | ≤0.25 | Aspirational optimization target from architecture |
| Ideal RTF | ≤0.10 | Aspirational. |
| Soak | 30–60 min → 2h MVP; earlier 8/24/72h ideas | Scale duration as product matures; no fake “passed |
| WER regression | Earlier <2 absolute points idea | Exploratory; TGFP prefers direct critical-content  |
| Model package | Earlier <1GB target | Product-design aspiration for ordinary endpoint. |
| Runtime RAM | Earlier <2GB | Aspiration; must benchmark actual student. |
| Average CPU | Earlier <30–35% | Aspiration; actual BPO device acceptance decides. |


## TABLE 50 (6 rows)

| Positioning | Value | Tradeoff |
| Per-seat on-device accent intelligence | Privacy + predictable subscription + low central c | Hard endpoint optimization R&D. |
| On-prem GPU gateway | Fastest route from high-quality GPU student to rea | Server deployment/concurrency/support. |
| Managed appliance | Simple enterprise support/procurement boundary. | Hardware operations and capital cost. |
| Cloud API/OEM | Easy integration for software partners. | Network/privacy/GPU economics; per-minute price pr |
| Custom accent/data/model service | High-value enterprise projects and proprietary cor | Services-heavy; must avoid fragmenting core model. |


## TABLE 51 (19 rows)

| Risk / unknown | Impact | Current mitigation / next evidence |
| Valid target cannot preserve identity | Fatal to current supervision thesis. | Gate 0 one-speaker/one-sentence; human same-person |
| Accent and identity are entangled | Major. | Natural cross-accent gold, multi-encoder identity  |
| Teacher reintroduces source accent or target voice | Major. | Measure accent after identity transfer; compare St |
| Existing realtime VC changes timbre, not pronuncia | Major. | Same-speaker paired fine-tune experiment; abort en |
| Content corruption: numbers/names/dates | Catastrophic in BPO. | Human critical-entity gate + WER/CER. |
| Already-correct speech damaged | High adoption/safety risk. | Per-token ALREADY-TARGET labels, damage curve, con |
| Natural cross-accent change requires rhythm/intona | Could invalidate v1 linguistic contract. | Gate-2 code-switcher timing/rhythm/F0 measurements |
| Offline quality needs too much lookahead | Can kill live product. | Quality/lookahead frontier only after offline qual |
| Endpoint CPU too slow | Commercial risk. | Existing realtime engines first; distill/quantize; |
| Server GPU concurrency too low | Unit economics risk. | Single-L4 benchmark before capacity purchase. |
| Licenses block target generation/distillation | Legal/commercial risk. | Dataset/model provenance manifests; replace NC/inf |
| Agent dislikes transformed identity | Commercial/adoption risk. | Agent acceptance is a pilot metric, not an afterth |
| BPO rejects price | Business risk. | Per-seat pilot with ROI metrics; benchmark against |
| Cross-platform complexity distracts Phase 0 | Schedule risk. | Design interfaces now; implement OS adapters only  |
| Infrastructure creates false progress | Execution risk. | Every sprint maps to one evidence milestone; no po |
| Random public dataset rights are invalid | Legal/data risk. | Original-source provenance, consent and license sn |
| Speaker embeddings leak accent | Model-design risk. | Do not assume speaker embedding is accent-invarian |
| Synthetic paired corpus amplifies teacher defects | Data-quality risk. | Teacher headroom; human validation; fine-tune on r |


## TABLE 53 (8 rows)

| Milestone | Direct evidence required |
| 1 — Valid Target | Same person, correct words, intended pronunciation |
| 2 — Learned Transformation | Neural model reproduces direction on held-out spee |
| 3 — Generalization | Unseen speakers improve without unacceptable conte |
| 4 — Streaming | Incremental conversion while speaker is still talk |
| 5 — Real-Time CPU | Runs fast enough on ordinary endpoint-class hardwa |
| 6 — Endpoint | Virtual microphone works in real calling applicati |
| 7 — Pilot | Real agents can use it and BPO wants to continue/p |


## TABLE 54 (16 rows)

| When / branch | Decision or finding |
| Earlier project | AccentEdge began as an on-device/direct S2S call-c |
| Legacy bake-off | Hundreds of tests and architecture scaffolding pro |
| Stage-A audit | Source/target provenance and evaluation were quest |
| Seed-VC branch | A real offline Seed-VC integration produced WAV ou |
| FACodec branch | Factorized architecture investigated. Critical tim |
| Fresh-start source of truth | Project reset to “one valid same-speaker target fi |
| Local model scan | CosyAccent, TokAN, FACodec, seq2seq-vc, PPG2PPG, C |
| On-prem sizing discussion | L4/L40S/RTX PRO tiers and RTF/latency concepts wer |
| Business concern | $1/min cloud service appeared difficult; Sanas pro |
| Business pivot | Per-seat/local-first enterprise model preferred; c |
| Production stack | Python/PyTorch training; later C++/ONNX native run |
| Cross-platform correction | Windows remains first BPO target, but architecture |
| Realtime shortcut | Before custom FACodec/Conformer, test RVC/Beatrice |
| Dataset scan | Large public source/target speech is available; co |
| Master consolidation | All branches reconciled into this dossier with sta |

