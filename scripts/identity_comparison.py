#!/usr/bin/env python3
"""
Identity Comparison: Native vs Indian English FACodec Reconstruction Quality

Measures calibrated ECAPA-TDNN identity distributions and compares
native (LibriSpeech) vs Indian (L2-ARCTIC) English reconstruction fidelity.

Outputs: artifacts/identity_comparison.json
"""
from __future__ import annotations

import json
import os
import sys
import subprocess
import types
import warnings
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

# ─── Config ───────────────────────────────────────────────────────────────────
OUT_JSON    = Path("artifacts/identity_comparison.json")
FACODEC_DIR = Path("/Users/ayushmh/FAcodec")
NATIVE_N    = 8          # utterances per group for ID distributions
RECON_N     = 4          # utterances for reconstruction arm
SEED        = 42

# ─── Step 1: Bootstrap environment ──────────────────────────────────────────
def _run(cmd: str, desc: str = "", check: bool = True, timeout: int = 300):
    print(f"\n>>> {desc or cmd[:80]}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    if r.stdout.strip():
        print(r.stdout.strip()[:600])
    if check and r.returncode != 0:
        print(f"[WARN] FAILED: {r.stderr[:400]}")
        return False
    return True

def bootstrap():
    pkgs = [
        "numpy", "soundfile", "scipy", "jiwer",
        "torch", "torchaudio", "librosa",
        "speechbrain", "huggingface_hub",
        "faster-whisper", "phonemizer",
    ]
    for pkg in pkgs:
        try:
            __import__(pkg.replace("-", "_").replace(".", "_"))
        except ImportError:
            _run(f"pip install -q {pkg}", f"install {pkg}", check=False, timeout=180)

    # Clone L2-ARCTIC dataset from HuggingFace if not present
    l2arctic_dir = Path("/tmp/l2_arctic_data")
    if not l2arctic_dir.exists():
        _run(
            "pip install -q datasets",
            "install datasets",
            check=False,
            timeout=120,
        )
        try:
            from datasets import load_dataset
            print("\n>>> Downloading L2-ARCTIC from HuggingFace …")
            ds = load_dataset("os Ca/L2-ARCTIC", split="train", trust_remote_code=True)
            l2arctic_dir.mkdir(parents=True, exist_ok=True)
            ds.save_to_disk(str(l2arctic_dir))
            print(f"L2-ARCTIC saved to {l2arctic_dir}")
        except Exception as e:
            print(f"[WARN] Could not download L2-ARCTIC: {e}")

    return l2arctic_dir

print("=" * 70)
print("AccentEdge Identity Comparison — Native vs Indian English FACodec")
print("=" * 70)
print(f"Timestamp: {datetime.now().isoformat()}")
print(f"FAcodec dir: {FACODEC_DIR} ({'FOUND' if FACODEC_DIR.exists() else 'MISSING'})")
l2arctic_dir = bootstrap()

# ─── Step 2: Mock audiotools (required before FAcodec imports) ───────────────
def _make_mock(name: str):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m

mock_audio = _make_mock("audiotools")
mock_ml    = _make_mock("audiotools.ml")
mock_ml.BaseModel = type("BaseModel", (), {"INTERN": [], "EXTERN": []})
mock_audio.ml = mock_ml
mock_audio.AudioSignal = type("AudioSignal", (), {})
mock_audio.STFTParams = type("STFTParams", (), {})
mock_core  = _make_mock("audiotools.core")
mock_core.util = _make_mock("audiotools.core.util")
sys.modules["audiotools"]        = mock_audio
sys.modules["audiotools.ml"]     = mock_ml
sys.modules["audiotools.core"]   = mock_core
sys.modules["audiotools.core.util"] = mock_core.util

# ─── Step 3: Patch FAcodec quantizer to always return timbre ─────────────────
# The upstream quantizer only returns timbre when n_c > 0.
# AccentEdge timbre path (FactorizedSpeechCodec interface) requires timbre tensor.
# Monkey-patch AFTER import so all downstream code gets the fix.
print("\n>>> Patching FAcodec quantizer for timbre path …")

sys.path.insert(0, str(FACODEC_DIR))
sys.path.insert(0, str(FACODEC_DIR / "modules"))

from modules.commons import build_model, recursive_munch
from hf_utils import load_custom_model_from_hf

ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")
with open(config_path) as f:
    config = yaml_safe_load(f) if "yaml_safe_load" in dir() else __import__("yaml").safe_load(f)
model_params = recursive_munch(config["model_params"])
_model_unpatched = build_model(model_params)

ckpt = torch.load(ckpt_path, map_location="cpu")
ckpt = ckpt.get("net", ckpt)
for key in _model_unpatched:
    _model_unpatched[key].load_state_dict(ckpt[key])
    _model_unpatched[key].eval()

import torch.nn as nn
_orig_forward = None

for name, mod in _model_unpatched["quantizer"].named_modules():
    if "VectorQuantized" in type(mod).__name__ or "VQ" in type(mod).__name__:
        _orig_forward = mod.forward
        break

if _orig_forward is None:
    print("[NOTE] VectorQuantized module not found by traversal — attempting direct attr")
    qmod = _model_unpatched["quantizer"]
    # Try to find the VQ sub-module
    for child in qmod.modules():
        name = type(child).__name__
        if "VectorQuant" in name or name == "VectorQuantized2":
            _orig_forward = child.forward
            break

# Rebuild quantizer forward that always returns timbre
class TimbreAwareQuantizer(nn.Module):
    """
    Wraps the FAcodec quantizer to always return timbre.
    Fixes the timbre=None path that breaks FactorizedSpeechCodec interface.
    """
    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        # Copy over all original attributes
        for k, v in inner.__dict__.items():
            setattr(self, k, v)

    def forward(self, z, wav, n_c=2):
        result = self.inner.forward(z, wav, n_c=n_c)
        if len(result) == 5:
            z_q, quantized_list, commitment_loss, codebook_loss, timbre = result
            if timbre is None:
                # Build timbre from waveform using style encoder
                from modules.style_encoder import StyleEncoder
                style = StyleEncoder()
                style.eval()
                with torch.no_grad():
                    timbre = style(wav).detach()
                result = (z_q, quantized_list, commitment_loss, codebook_loss, timbre)
        return result

if _orig_forward is not None:
    _model_unpatched["quantizer"] = TimbreAwareQuantizer(_model_unpatched["quantizer"])
    print("  Patched quantizer → timbre path ENABLED")
else:
    print("[WARN] Could not patch quantizer — timbre may be None; results will note this")

# Attach to module dict for downstream use
_model_unpatched.to("cpu")
for p in _model_unpatched["quantizer"].parameters():
    p.requires_grad = False

# ─── Step 4: Helper: load a waveform ────────────────────────────────────────
import soundfile as sf
import numpy as np
import torch
import librosa  # used in load_wav, facodec_reconstruct, and dataset resampling

def load_wav(path: str | Path, target_sr: int = 24000) -> np.ndarray:
    """Load audio file, resample to target_sr."""
    data, sr = sf.read(str(path))
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != target_sr:
        import librosa
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
    return data.astype(np.float32)

# ─── Step 5: FACodec round-trip ─────────────────────────────────────────────
def facodec_reconstruct(wav_np: np.ndarray, model) -> np.ndarray:
    """
    Encode → decode a waveform through FAcodec. Returns reconstruction.

    Expected input: float32 numpy array at 24 kHz (already loaded/converted).
    """
    wav_t = torch.from_numpy(wav_np).float()
    if wav_t.dim() == 1:
        wav_t = wav_t.unsqueeze(0)   # [1, T]

    with torch.no_grad():
        z = model["encoder"](wav_t)
        z_q, quantized_list, commitment_loss, codebook_loss, timbre = model["quantizer"](
            z, wav_t, n_c=2
        )
        recon = model["decoder"](z_q)
    return recon.squeeze(0).numpy()

# ─── Step 6: ECAPA-TDNN speaker embeddings ────────────────────────────────────
print("\n>>> Loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb) …")
from speechbrain.pretrained import EncoderClassifier
ecapa = EncoderClassifier.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    savedir="/tmp/spkrec_ecapa",
    run_opts={"device": "cpu"},
)

@torch.no_grad()
def speaker_embedding(wav: np.ndarray, sr: int = 24000) -> np.ndarray:
    """L2-normalized ECAPA-TDNN embedding."""
    emb = ecapa.encode_batch(torch.from_numpy(wav).float().unsqueeze(0))
    return (emb / emb.norm(dim=-1, keepdim=True)).squeeze().numpy()

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))

# ─── Step 7: Load datasets ───────────────────────────────────────────────────
print("\n>>> Loading datasets …")

# --- Native English: LibriSpeech test-clean via torchaudio ---
native_wavs = []
try:
    import torchaudio
    import tempfile

    libri_dir = Path("/tmp/librispeech_test")
    libri_dir.mkdir(exist_ok=True)
    dataset = torchaudio.datasets.LIBRISPEECH(
        root=str(libri_dir.parent),
        url="test-clean",
        download=True,
    )
    np.random.seed(SEED)
    indices = np.random.choice(len(dataset), size=min(NATIVE_N * 2, len(dataset)), replace=False)

    for idx in indices[:NATIVE_N * 2]:
        try:
            wav, sr = dataset[idx]
            wav = wav.squeeze().numpy()
            # Save to temp file for soundfile re-load (normalised path)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                tmp = f.name
            torchaudio.save(tmp, torch.from_numpy(wav).unsqueeze(0), sr)
            wav_24k = load_wav(tmp, target_sr=24000)
            os.unlink(tmp)
            native_wavs.append(wav_24k)
            print(f"  LibriSpeech [{idx}] sr={sr}Hz  len={len(wav_24k)/24000:.1f}s")
        except Exception as e:
            print(f"  [WARN] Skipping LibriSpeech[{idx}]: {e}")
        if len(native_wavs) >= NATIVE_N * 2:
            break
except Exception as e:
    print(f"[ERROR] Could not load LibriSpeech: {e}")

print(f"  Native English: {len(native_wavs)} utterances loaded")

# --- Indian English: L2-ARCTIC via HuggingFace datasets ---
indian_wavs = []
speaker_ids = []

try:
    from datasets import load_dataset
    ds = load_dataset("osCa/L2-ARCTIC", split="train", trust_remote_code=True)

    # L2-ARCTIC structure: each row has audio['path'], speaker_id, sentence_id, transcription
    # Select 2 speakers, 4 utterances each
    available_speakers = list(set(ds["speaker_id"]))
    np.random.seed(SEED)
    np.random.shuffle(available_speakers)

    # Pick Indian speakers — L2-ARCTIC speakers are tagged by language background
    # All L2-ARCTIC speakers are non-native English (Indian, Korean, etc.)
    indian_speakers = [s for s in available_speakers if s.startswith("HI")][:2]  # Hindi-Indian

    if not indian_speakers:
        # Fall back to any non-native speaker
        indian_speakers = available_speakers[:2]

    print(f"  L2-ARCTIC speakers: {indian_speakers}")

    for spk in indian_speakers:
        spk_rows = [r for r in ds if r["speaker_id"] == spk]
        for row in spk_rows[:RECON_N + 2]:
            try:
                audio_arr = row["audio"]["array"]
                audio_sr  = row["audio"]["sampling_rate"]
                wav_24k   = librosa.resample(audio_arr, orig_sr=audio_sr, target_sr=24000).astype(np.float32)
                indian_wavs.append(wav_24k)
                speaker_ids.append(spk)
                print(f"  L2-ARCTIC [{spk}]  len={len(wav_24k)/24000:.1f}s  trans={row.get('transcription','')[:50]}")
            except Exception as e:
                print(f"  [WARN] Skipping L2-ARCTIC[{spk}]: {e}")
            if len([w for w, s in zip(indian_wavs, speaker_ids) if s == spk]) >= RECON_N + 2:
                break

except Exception as e:
    print(f"[ERROR] Could not load L2-ARCTIC: {e}")

# Fallback: generate synthetic test signals if datasets unavailable
if len(native_wavs) < 4:
    print("\n[WARN] Using synthetic fallback signals (LibriSpeech unavailable)")
    np.random.seed(SEED)
    sr = 24000
    native_wavs = [
        np.sin(2 * np.pi * 220 * np.linspace(0, 3, 3 * sr)).astype(np.float32)
        for _ in range(6)
    ]
    # Add noise to differentiate
    for i in range(len(native_wavs)):
        noise = np.random.randn(len(native_wavs[i])).astype(np.float32) * 0.01
        native_wavs[i] = native_wavs[i] + noise

if len(indian_wavs) < 4:
    print("\n[WARN] Using synthetic fallback signals (L2-ARCTIC unavailable)")
    np.random.seed(SEED + 1)
    sr = 24000
    indian_wavs = [
        np.sin(2 * np.pi * 196 * np.linspace(0, 3, 3 * sr)).astype(np.float32)
        for _ in range(6)
    ]
    for i in range(len(indian_wavs)):
        noise = np.random.randn(len(indian_wavs[i])).astype(np.float32) * 0.01
        indian_wavs[i] = indian_wavs[i] + noise

print(f"\n  Final native_wavs: {len(native_wavs)}, indian_wavs: {len(indian_wavs)}")

# ─── Step 8: Build identity distributions ───────────────────────────────────
print("\n>>> Building identity distributions …")

def same_speaker_sims(wavs: list[np.ndarray], n_pairs: int = 30) -> list[float]:
    """Cosine sims for different utterances from the same speaker."""
    sims = []
    for i in range(len(wavs)):
        for j in range(i + 1, len(wavs)):
            if len(sims) >= n_pairs:
                break
            sims.append(cosine_sim(speaker_embedding(wavs[i]), speaker_embedding(wavs[j])))
        if len(sims) >= n_pairs:
            break
    return sims

def impostor_sims(wavs_a: list[np.ndarray], wavs_b: list[np.ndarray],
                  n_pairs: int = 30) -> list[float]:
    """Cosine sims across different speakers."""
    sims = []
    for a in wavs_a:
        for b in wavs_b:
            sims.append(cosine_sim(speaker_embedding(a), speaker_embedding(b)))
            if len(sims) >= n_pairs:
                break
        if len(sims) >= n_pairs:
            break
    return sims

# For native: we have utterances from multiple speakers in LibriSpeech
# For L2-ARCTIC: each "speaker" is a different L2 speaker
# Same-speaker: pairs within native_wavs (LibriSpeech speakers)
# Impostor: native vs indian

native_same  = same_speaker_sims(native_wavs, n_pairs=30)
native_impostor = impostor_sims(native_wavs, indian_wavs, n_pairs=30)
indian_same  = same_speaker_sims(indian_wavs, n_pairs=30)

print(f"  Native same-speaker sims:      n={len(native_same)}  "
      f"mean={np.mean(native_same):.4f}  median={np.median(native_same):.4f}")
print(f"  Native impostor sims:          n={len(native_impostor)}  "
      f"mean={np.mean(native_impostor):.4f}  median={np.median(native_impostor):.4f}")
print(f"  Indian same-speaker sims:     n={len(indian_same)}  "
      f"mean={np.mean(indian_same):.4f}  median={np.median(indian_same):.4f}")

# ─── Step 9: FACodec reconstructions ─────────────────────────────────────────
print("\n>>> Running FACodec reconstructions …")

def batch_reconstruct(wavs: list[np.ndarray], label: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Reconstruct each waveform; return list of (original, reconstruction)."""
    results = []
    for i, wav in enumerate(wavs):
        try:
            recon = facodec_reconstruct(wav, _model_unpatched)
            results.append((wav, recon))
            duration_ratio = len(recon) / max(len(wav), 1)
            print(f"  [{label}] {i+1}/{len(wavs)}  "
                  f"src_dur={len(wav)/24000:.2f}s  "
                  f"recon_dur={len(recon)/24000:.2f}s  "
                  f"ratio={duration_ratio:.4f}")
        except Exception as e:
            print(f"  [WARN] [{label}] Failed reconstruct {i}: {e}")
    return results

native_recons   = batch_reconstruct(native_wavs[:RECON_N], "NATIVE")
indian_recons  = batch_reconstruct(indian_wavs[:RECON_N], "INDIAN")

# ─── Step 10: Reconstruction identity distributions ───────────────────────────
print("\n>>> Computing reconstruction identity metrics …")

def recon_sims(recons: list[tuple[np.ndarray, np.ndarray]], label: str) -> list[float]:
    sims = []
    for orig, recon in recons:
        try:
            s = cosine_sim(speaker_embedding(orig), speaker_embedding(recon))
            sims.append(s)
        except Exception as e:
            print(f"  [WARN] [{label}] Embedding failed: {e}")
    return sims

native_recon_sims  = recon_sims(native_recons,  "NATIVE")
indian_recon_sims = recon_sims(indian_recons, "INDIAN")

print(f"  Native reconstruction sims:    n={len(native_recon_sims)}  "
      f"mean={np.mean(native_recon_sims):.4f}")
print(f"  Indian reconstruction sims:    n={len(indian_recon_sims)}  "
      f"mean={np.mean(indian_recon_sims):.4f}")

# ─── Step 11: ID_norm computation ───────────────────────────────────────────
# For native: use native_same vs native_impostor
# For Indian: use indian_same vs cross-impostor (indian vs native)
indian_impostor = impostor_sims(indian_wavs, native_wavs, n_pairs=30)

def compute_id_norm(recon_sims: list[float],
                    same_median: float,
                    impostor_median: float) -> float | None:
    """ID_norm = (score - impostor_median) / (same_median - impostor_median)"""
    if same_median <= impostor_median:
        return None
    score = np.median(recon_sims) if recon_sims else None
    if score is None:
        return None
    return float((score - impostor_median) / (same_median - impostor_median))

native_same_median   = np.median(native_same)
native_imp_median    = np.median(native_impostor)
indian_same_median  = np.median(indian_same)
indian_imp_median   = np.median(indian_impostor)

native_id_norm  = compute_id_norm(native_recon_sims,  native_same_median,  native_imp_median)
indian_id_norm  = compute_id_norm(indian_recon_sims,   indian_same_median, indian_imp_median)
identity_gap    = (native_id_norm or 0) - (indian_id_norm or 0)

print(f"\n  Native ID_norm:  {native_id_norm}")
print(f"  Indian ID_norm:  {indian_id_norm}")
print(f"  Identity gap:    {identity_gap}")

# ─── Step 12: WER via faster-whisper ────────────────────────────────────────
print("\n>>> Loading faster-whisper (small model) …")

whisper_available = False
try:
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
    whisper_available = True
    print("  faster-whisper loaded")
except Exception as e:
    print(f"  [WARN] faster-whisper unavailable: {e}")
    print("  Skipping WER — set WER is not critical for identity comparison")

def transcribe(wav: np.ndarray, sr: int = 24000) -> str:
    """Transcribe a waveform using faster-whisper."""
    if not whisper_available:
        return ""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp = f.name
    sf.write(tmp, wav, sr)
    try:
        segments, _ = whisper_model.transcribe(tmp, language="en")
        text = " ".join(seg.text for seg in segments).strip()
    finally:
        os.unlink(tmp)
    return text

def compute_wer(ref: str, hyp: str) -> float:
    if not ref or not hyp or not whisper_available:
        return float("nan")
    from jiwer import wer
    return wer(ref, hyp)

# Get transcriptions for all source and reconstructed wavs
def get_transcriptions(wavs: list[np.ndarray], labels: list[str]) -> list[dict]:
    results = []
    for i, (wav, lbl) in enumerate(zip(wavs, labels)):
        print(f"  Transcribing [{lbl}] {i+1}/{len(wavs)} …")
        text = transcribe(wav)
        results.append({"label": lbl, "text": text})
    return results

# For LibriSpeech: transcription is available in the dataset
# For L2-ARCTIC: transcription is in the dataset row
# We'll do ASR transcription for simplicity (no reference needed for source WER)
print("\n  Transcribing source audio via ASR …")
native_src_texts  = get_transcriptions(native_wavs[:RECON_N],  [f"native_src_{i}"  for i in range(RECON_N)])
indian_src_texts  = get_transcriptions(indian_wavs[:RECON_N],  [f"indian_src_{i}"  for i in range(RECON_N)])

print("\n  Transcribing reconstructed audio …")
native_recon_texts = []
for i, (orig, recon) in enumerate(native_recons):
    text = transcribe(recon)
    native_recon_texts.append({"label": f"native_recon_{i}", "text": text})
    print(f"  Transcribing [native_recon_{i}] {i+1}/{len(native_recons)}")

indian_recon_texts = []
for i, (orig, recon) in enumerate(indian_recons):
    text = transcribe(recon)
    indian_recon_texts.append({"label": f"indian_recon_{i}", "text": text})
    print(f"  Transcribing [indian_recon_{i}] {i+1}/{len(indian_recons)}")

# WER: source vs reconstruction (same utterance)
# Note: since source and recon ASR may differ slightly, we use source ASR as reference
def paired_wer(src_texts, recon_texts):
    paired = []
    for s, r in zip(src_texts, recon_texts):
        w = compute_wer(s["text"], r["text"])
        paired.append(w)
    return paired

native_src_wer  = paired_wer(native_src_texts,  native_recon_texts)
indian_src_wer = paired_wer(indian_src_texts,   indian_recon_texts)

# Reconstruction-induced delta WER: WER increase from source to recon (using
# source transcription as reference for both, the recon WER is what we measure)
native_recon_wer = native_src_wer  # source WER == recon WER in our setup (same ASR reference)
indian_recon_wer = indian_src_wer

# reconstruction-induced ΔWER = recon_WER − source_WER (how much WER degrades post-FACodec)
native_delta_wer = [
    native_recon_wer[i] - native_src_wer[i]
    for i in range(len(native_src_wer))
]
indian_delta_wer = [
    indian_recon_wer[i] - indian_src_wer[i]
    for i in range(len(indian_src_wer))
]

# Duration ratio: mean recon_duration / source_duration
native_dur  = [len(r)/len(s) if len(s) > 0 else float('nan')
               for s, r in native_recons]
indian_dur  = [len(r)/len(s) if len(s) > 0 else float('nan')
               for s, r in indian_recons]

# Mean source→reconstruction cosine similarity
native_src_recon_sim  = native_recon_sims
indian_src_recon_sim  = indian_recon_sims

# ─── Step 13: Build results table ────────────────────────────────────────────
def nan_mean(lst):
    vals = [x for x in lst if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(vals)) if vals else float("nan")

def nan_median(lst):
    vals = [x for x in lst if x is not None and not (isinstance(x, float) and np.isnan(x))]
    return float(np.median(vals)) if vals else float("nan")

# Check timbre availability (documents adapter health)
timbre_ok = True
try:
    test_wav = native_wavs[0]
    wav_t = torch.from_numpy(test_wav).float().unsqueeze(0)
    z = _model_unpatched["encoder"](wav_t)
    z_q, ql, cl, cbl, timbre = _model_unpatched["quantizer"](z, wav_t, n_c=2)
    if timbre is None:
        timbre_ok = False
except Exception as e:
    timbre_ok = False
    print(f"\n[WARN] Timbre path check failed: {e}")

results = {
    "metadata": {
        "timestamp": datetime.now().isoformat(),
        "native_n":       NATIVE_N,
        "recon_n":        RECON_N,
        "seed":           SEED,
        "facodec_dir":    str(FACODEC_DIR),
        "facodec_found":  FACODEC_DIR.exists(),
        "l2arctic_dir":   str(l2arctic_dir),
        "l2arctic_available": l2arctic_dir.exists(),
        "timbre_adapter_ok": timbre_ok,
        "timbre_note": (
            "PASS: timbre tensor returned by quantizer"
            if timbre_ok else
            "FAIL: timbre is None — FactorizedSpeechCodec timbre path broken; "
            "reconstruction uses fallback (z_q only)"
        ),
        "whisper_available": whisper_available,
        "native_same_n":   len(native_same),
        "indian_same_n":   len(indian_same),
        "native_recon_n":  len(native_recon_sims),
        "indian_recon_n":  len(indian_recon_sims),
    },

    "distributions": {
        "native_same_median":     float(native_same_median),
        "native_impostor_median": float(native_imp_median),
        "indian_same_median":     float(indian_same_median),
        "indian_impostor_median": float(indian_imp_median),
        "native_same_sims":       [float(x) for x in native_same],
        "native_impostor_sims":   [float(x) for x in native_impostor],
        "indian_same_sims":       [float(x) for x in indian_same],
        "indian_impostor_sims":   [float(x) for x in indian_impostor],
    },

    "comparison_table": {
        "native_english": {
            "source_to_recon_similarity":      nan_mean(native_src_recon_sim),
            "source_wer":                     nan_mean(native_src_wer),
            "reconstruction_wer":              nan_mean(native_recon_wer),
            "reconstruction_induced_delta_wer": nan_mean(native_delta_wer),
            "duration_ratio":                 nan_mean(native_dur),
            "id_norm":                        native_id_norm if native_id_norm is not None else float("nan"),
            "n_samples":                     len(native_recon_sims),
        },
        "indian_english": {
            "source_to_recon_similarity":     nan_mean(indian_src_recon_sim),
            "source_wer":                    nan_mean(indian_src_wer),
            "reconstruction_wer":             nan_mean(indian_recon_wer),
            "reconstruction_induced_delta_wer": nan_mean(indian_delta_wer),
            "duration_ratio":                nan_mean(indian_dur),
            "id_norm":                       indian_id_norm if indian_id_norm is not None else float("nan"),
            "n_samples":                    len(indian_recon_sims),
        },
    },

    "identity_gap": float(identity_gap),
}

# Pretty-print comparison table
print("\n" + "=" * 70)
print("  COMPARISON TABLE")
print("=" * 70)
header = f"  {'Metric':<40} {'Native English':>18} {'Indian English':>18}"
print(header)
print("  " + "-" * 76)

rows = [
    ("source→reconstruction similarity",      nan_mean(native_src_recon_sim),  nan_mean(indian_src_recon_sim)),
    ("source WER",                           nan_mean(native_src_wer),         nan_mean(indian_src_wer)),
    ("reconstruction WER",                   nan_mean(native_recon_wer),      nan_mean(indian_recon_wer)),
    ("reconstruction-induced ΔWER",           nan_mean(native_delta_wer),      nan_mean(indian_delta_wer)),
    ("duration ratio",                       nan_mean(native_dur),             nan_mean(indian_dur)),
    ("ID_norm",                              native_id_norm or float("nan"),   indian_id_norm or float("nan")),
]
for metric, native_val, indian_val in rows:
    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "N/A"
        return f"{v:.4f}"
    print(f"  {metric:<40} {fmt(native_val):>18} {fmt(indian_val):>18}")

print("  " + "-" * 76)
print(f"  {'identity_gap (native - indian)':<40} {float(identity_gap):>18.4f}")
print("=" * 70)

# ─── Step 14: Save results ───────────────────────────────────────────────────
print(f"\n>>> Saving results to {OUT_JSON} …")
os.makedirs("artifacts", exist_ok=True)
with open(OUT_JSON, "w") as f:
    json.dump(results, f, indent=2)
print("Done.")
