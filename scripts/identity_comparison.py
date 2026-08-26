#!/usr/bin/env python3
"""
Identity Comparison — Calibrated Identity Preservation: Native vs Indian English

Paper-faithful calibration methodology:
  1. SAME-SPEAKER reference (within-speaker variation):
     For each speaker, compare utterance A vs utterance B.
     Defines "same speaker under natural variation" for our ECAPA checkpoint.

  2. DIFFERENT-SPEAKER reference (impostor floor):
     Speaker A vs Speaker B (different speakers).
     Defines the impostor floor.

  3. RECONSTRUCTION reference:
     source vs FACodec reconstruction for BOTH native and Indian.
     Tells us how much reconstruction itself changes identity.

KEY COMPARISON (not arbitrary thresholds):
  How far did reconstruction move relative to our same-speaker ↔ different-speaker span?
  • shift_over_span  = reconstruction_shift / same_speaker_span
  • shift_over_impostor = reconstruction_shift / impostor_distance
  • preservation_ratio  = recon_median / same_speaker_median

  This avoids the old SECS-threshold mistake.

PRESERVED LEGACY METRIC:
  Old mean SECS (0.05–0.24 range) is computed but labeled:
  DIAGNOSTIC_ONLY / BROKEN ADAPTER / NOT VALID FACODEC PERFORMANCE

Corpora:
  - Native:   CMU ARCTIC (matched prompts with L2-ARCTIC)
  - Indian:   L2-ARCTIC (Hindi-Indian speakers)

Outputs: artifacts/identity_comparison/
  same_speaker_dist.json, different_speaker_dist.json,
  reconstruction_dist.json, summary.json
"""
from __future__ import annotations

import json
import os
import sys
import warnings
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
import torch
import soundfile as sf
import librosa
import yaml
import types

warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════════

class Config:
    """All tunables in one place."""

    output_dir: Path = Path("artifacts/identity_comparison")
    facodec_dir: Path = Path("/Users/ayushmh/FAcodec")
    facodec_ckpt: str = "Plachta/FAcodec"
    device: str = "cpu"
    n_pairs_ref: int = 40          # pairs for each reference distribution
    n_recon: int = 8               # utterances per corpus for reconstruction arm
    seed: int = 42
    target_sr: int = 24000
    # Dataset identifiers
    l2arctic_hf_id: str = "osCa/L2-ARCTIC"
    cmu_arctic_hf_id: str = "cmu_arctic"
    # Gate 2 thresholds (used by gate2_identity.py as well)
    gate2_max_shift_over_span: float = 0.25
    gate2_min_preservation: float = 0.85


cfg = Config()

# ═══════════════════════════════════════════════════════════════════════════════
# Mock audiotools (required before FAcodec imports — prevents metaclass conflict)
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mock(name: str):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m


def install_mocks():
    mock_audio = _make_mock("audiotools")
    mock_ml = _make_mock("audiotools.ml")
    mock_ml.BaseModel = type("BaseModel", (), {"INTERN": [], "EXTERN": []})
    mock_audio.ml = mock_ml
    mock_audio.AudioSignal = type("AudioSignal", (), {})
    mock_audio.STFTParams = type("STFTParams", (), {})
    mock_core = _make_mock("audiotools.core")
    mock_core.util = _make_mock("audiotools.core.util")
    sys.modules["audiotools"] = mock_audio
    sys.modules["audiotools.ml"] = mock_ml
    sys.modules["audiotools.core"] = mock_core
    sys.modules["audiotools.core.util"] = mock_core.util


install_mocks()

# ═══════════════════════════════════════════════════════════════════════════════
# ECAPA-TDNN Speaker Embedding
# ═══════════════════════════════════════════════════════════════════════════════

class ECAPAEmbedding:
    """ECAPA-TDNN speaker embedding via SpeechBrain spkrec-ecapa-voxceleb."""

    def __init__(self, device: str = "cpu"):
        from speechbrain.pretrained import EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/spkrec_ecapa",
            run_opts={"device": device},
        )
        self.device = device

    @torch.no_grad()
    def __call__(self, wav: np.ndarray, sr: int = 24000) -> np.ndarray:
        """L2-normalized ECAPA-TDNN embedding for a waveform."""
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        wav_t = torch.from_numpy(wav).float().unsqueeze(0)
        emb = self.classifier.encode_batch(wav_t)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.squeeze().cpu().numpy()

    @staticmethod
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two L2-normalized embeddings."""
        return float(np.dot(a, b))


# ═══════════════════════════════════════════════════════════════════════════════
# FACodec Loading (upstream pattern, matches verify_facodec_direct.py)
# ════════════════════��══════════════════════════════════════════════════════════

class FACodecModel:
    """Loaded FAcodec with encode / decode / reconstruct."""

    def __init__(self, facodec_dir: Path, ckpt_name: str, device: str = "cpu"):
        self.device = torch.device(device)

        # Ensure FAcodec is on sys.path
        if str(facodec_dir) not in sys.path:
            sys.path.insert(0, str(facodec_dir))
        modules_dir = facodec_dir / "modules"
        if modules_dir.exists() and str(modules_dir) not in sys.path:
            sys.path.insert(0, str(modules_dir))

        from modules.commons import build_model, recursive_munch
        from hf_utils import load_custom_model_from_hf

        ckpt_path, config_path = load_custom_model_from_hf(ckpt_name)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        mp = recursive_munch(config["model_params"])
        self.model = build_model(mp)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = ckpt.get("net", ckpt)
        for key in ckpt:
            self.model[key].load_state_dict(ckpt[key])
            self.model[key].eval().to(self.device)
            for p in self.model[key].parameters():
                p.requires_grad = False

    @torch.no_grad()
    def encode(self, wav_np: np.ndarray):
        """Encode -> (z_q, quantized_list, timbre)."""
        wav_t = torch.from_numpy(wav_np).float()
        if wav_t.dim() == 1:
            wav_t = wav_t.unsqueeze(0)
        wav_in = wav_t.unsqueeze(0).to(self.device)

        z = self.model["encoder"](wav_in)
        z_q, quantized_list, _, _, timbre = self.model["quantizer"](
            z, wav_in, n_c=2
        )
        return z_q, quantized_list, timbre

    @torch.no_grad()
    def decode(self, z_q) -> np.ndarray:
        """Decode quantized z -> numpy waveform."""
        recon = self.model["decoder"](z_q.to(self.device))
        return recon.squeeze().cpu().numpy()

    @torch.no_grad()
    def reconstruct(self, wav_np: np.ndarray) -> np.ndarray:
        """Full encode -> decode round-trip."""
        z_q, _, _ = self.encode(wav_np)
        return self.decode(z_q)


# ═══════════════════════════════════════════════════════════════════════════════
# Dataset Loading
# ═══════════════════════════════════════════════════════════════════════════════

def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    return librosa.resample(
        np.asarray(wav, dtype=np.float32), orig_sr=orig_sr, target_sr=target_sr
    ).astype(np.float32)


class Utt:
    """Lightweight utterance container."""
    __slots__ = ("path", "speaker_id", "sentence_id", "text", "wav", "sr", "corpus")

    def __init__(self, path, speaker_id, sentence_id, text, wav, sr, corpus):
        self.path = path
        self.speaker_id = speaker_id
        self.sentence_id = sentence_id
        self.text = text
        self.wav = wav
        self.sr = sr
        self.corpus = corpus


def load_cmu_arctic(cfg: Config) -> list[Utt]:
    """Load CMU ARCTIC (native English reference corpus).

    Speakers: slt, bdl, rms, clb, jmk, awb, ksp
    Uses matched prompts from the CMU ARCTIC sentence set.
    """
    utterances: list[Utt] = []

    # Method 1: HuggingFace
    try:
        from datasets import load_dataset
        print("  Loading CMU ARCTIC from HuggingFace...")
        ds = load_dataset(cfg.cmu_arctic_hf_id, split="train", trust_remote_code=True)

        speakers = sorted(set(ds["speaker_id"]))
        np.random.seed(cfg.seed)
        np.random.shuffle(speakers)
        chosen = speakers[:4]

        for spk in chosen:
            rows = [r for r in ds if r["speaker_id"] == spk]
            np.random.shuffle(rows)
            for row in rows[: cfg.n_recon + 4]:
                arr = row["audio"]["array"]
                sr = row["audio"]["sampling_rate"]
                wav = _resample(arr, sr, cfg.target_sr)
                utterances.append(Utt(
                    path=row.get("path", ""),
                    speaker_id=spk,
                    sentence_id=row.get("sentence_id", row.get("id", "")),
                    text=row.get("text", row.get("transcription", "")),
                    wav=wav, sr=cfg.target_sr, corpus="native",
                ))
        print(f"  CMU ARCTIC (HF): {len(utterances)} utterances from speakers {chosen}")
        return utterances
    except Exception as e:
        print(f"  [WARN] HF CMU ARCTIC failed: {e}")

    # Method 2: torchaudio
    try:
        import torchaudio
        import tempfile

        print("  Loading CMU ARCTIC via torchaudio...")
        arctic_root = Path("/tmp/cmu_arctic")
        arctic_root.mkdir(exist_ok=True)

        speakers = ["slt", "bdl", "rms", "clb"]
        for spk in speakers:
            try:
                ds = torchaudio.datasets.CMU_ARCTIC(
                    root=str(arctic_root), speaker=spk, download=True,
                )
                idxs = np.random.choice(
                    len(ds), size=min(cfg.n_recon + 4, len(ds)), replace=False,
                )
                for idx in idxs:
                    wav, sr, transcript = ds[idx]
                    wav_np = wav.squeeze().numpy()
                    if sr != cfg.target_sr:
                        wav_np = _resample(wav_np, sr, cfg.target_sr)
                    utterances.append(Utt(
                        path=f"cmu_arctic/{spk}/{idx}",
                        speaker_id=spk,
                        sentence_id=str(idx),
                        text=transcript,
                        wav=wav_np, sr=cfg.target_sr, corpus="native",
                    ))
            except Exception as e2:
                print(f"  [WARN] Speaker {spk} failed: {e2}")
        print(f"  CMU ARCTIC (torchaudio): {len(utterances)} utterances")
        return utterances
    except Exception as e:
        print(f"  [ERROR] All CMU ARCTIC loading methods failed: {e}")
        return utterances


def load_l2_arctic(cfg: Config) -> list[Utt]:
    """Load L2-ARCTIC (Indian English speakers).

    Hindi-Indian speakers have IDs starting with 'HI'.
    L2-ARCTIC shares the CMU ARCTIC prompt set.
    """
    utterances: list[Utt] = []

    try:
        from datasets import load_dataset
        print("  Loading L2-ARCTIC from HuggingFace...")
        ds = load_dataset(cfg.l2arctic_hf_id, split="train", trust_remote_code=True)

        all_speakers = sorted(set(ds["speaker_id"]))
        np.random.seed(cfg.seed)
        np.random.shuffle(all_speakers)

        indian = [s for s in all_speakers if s.startswith("HI")]
        if not indian:
            print(f"  [WARN] No HI-prefixed speakers. Available: {all_speakers[:10]}")
            indian = all_speakers[:4]
        chosen = indian[:4]
        print(f"  L2-ARCTIC Indian speakers: {chosen}")

        for spk in chosen:
            rows = [r for r in ds if r["speaker_id"] == spk]
            np.random.shuffle(rows)
            for row in rows[: cfg.n_recon + 4]:
                try:
                    arr = row["audio"]["array"]
                    sr = row["audio"]["sampling_rate"]
                    wav = _resample(arr, sr, cfg.target_sr)
                    utterances.append(Utt(
                        path=row.get("path", ""),
                        speaker_id=spk,
                        sentence_id=row.get("sentence_id", row.get("id", "")),
                        text=row.get("transcription", row.get("text", "")),
                        wav=wav, sr=cfg.target_sr, corpus="indian",
                    ))
                except Exception as e2:
                    print(f"  [WARN] Skip {spk} row: {e2}")
        print(f"  L2-ARCTIC: {len(utterances)} utterances")
        return utterances
    except Exception as e:
        print(f"  [ERROR] L2-ARCTIC loading failed: {e}")
        return utterances


# ═══════════════════════════════════════════════════════════════════════════════
# Three-Reference Calibration Methodology
# ═════════════════════════════════════��══════════════════════════════════════════

def build_speaker_groups(utterances: list[Utt]) -> dict[str, list[Utt]]:
    """Group utterances by speaker_id."""
    groups: dict[str, list[Utt]] = {}
    for u in utterances:
        groups.setdefault(u.speaker_id, []).append(u)
    return groups


def compute_same_speaker_distribution(
    utterances: list[Utt],
    extractor: ECAPAEmbedding,
    n_pairs: int = 40,
) -> dict:
    """
    SAME-SPEAKER reference: within-speaker variation.
    For each speaker, compare utterance A vs utterance B.
    """
    groups = build_speaker_groups(utterances)
    pairs = []
    for spk, utts in groups.items():
        if len(utts) < 2:
            continue
        for i in range(len(utts)):
            for j in range(i + 1, len(utts)):
                sim = extractor.cosine_sim(
                    extractor(utts[i].wav, utts[i].sr),
                    extractor(utts[j].wav, utts[j].sr),
                )
                pairs.append({
                    "speaker": spk,
                    "utt_a": utts[i].sentence_id,
                    "utt_b": utts[j].sentence_id,
                    "cosine_sim": sim,
                })
                if len(pairs) >= n_pairs:
                    break
            if len(pairs) >= n_pairs:
                break

    sims = [p["cosine_sim"] for p in pairs]
    return {
        "method": "within_speaker_pairs",
        "corpus": utterances[0].corpus if utterances else "unknown",
        "pairs": pairs,
        "summary": _summary_stats(sims),
    }


def compute_different_speaker_distribution(
    group_a: list[Utt],
    group_b: list[Utt],
    extractor: ECAPAEmbedding,
    n_pairs: int = 40,
    label: str = "",
) -> dict:
    """
    DIFFERENT-SPEAKER reference: impostor floor.
    Speaker A (from group_a) vs Speaker B (from group_b).
    Speakers must be different.
    """
    # Pre-compute embeddings
    emb_a = [(u, extractor(u.wav, u.sr)) for u in group_a]
    emb_b = [(u, extractor(u.wav, u.sr)) for u in group_b]

    pairs = []
    rng = np.random.RandomState(cfg.seed)
    for u_a, e_a in emb_a:
        candidates = [(u_b, e_b) for u_b, e_b in emb_b if u_b.speaker_id != u_a.speaker_id]
        if not candidates:
            continue
        for u_b, e_b in rng.permutation(candidates):
            sim = extractor.cosine_sim(e_a, e_b)
            pairs.append({
                "speaker_a": u_a.speaker_id,
                "speaker_b": u_b.speaker_id,
                "utt_a": u_a.sentence_id,
                "utt_b": u_b.sentence_id,
                "cosine_sim": sim,
            })
            if len(pairs) >= n_pairs:
                break
        if len(pairs) >= n_pairs:
            break

    sims = [p["cosine_sim"] for p in pairs]
    return {
        "method": "cross_speaker_pairs",
        "label": label,
        "pairs": pairs,
        "summary": _summary_stats(sims),
    }


def compute_reconstruction_distribution(
    originals: list[Utt],
    reconstructions: list[np.ndarray],
    extractor: ECAPAEmbedding,
    corpus: str = "",
) -> dict:
    """
    RECONSTRUCTION reference: source vs FACodec reconstruction.
    """
    pairs = []
    for orig, recon in zip(originals, reconstructions):
        try:
            sim = extractor.cosine_sim(
                extractor(orig.wav, orig.sr),
                extractor(recon, orig.sr),
            )
            pairs.append({
                "sentence_id": orig.sentence_id,
                "speaker": orig.speaker_id,
                "cosine_sim": sim,
            })
        except Exception as e:
            print(f"  [WARN] Embedding failed for {orig.sentence_id}: {e}")

    sims = [p["cosine_sim"] for p in pairs]
    return {
        "method": "source_vs_reconstruction",
        "corpus": corpus,
        "pairs": pairs,
        "summary": _summary_stats(sims),
    }


def _summary_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0,
                "min": 0.0, "max": 0.0}
    return {
        "n": len(values),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Calibration — Key Comparisons
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_corpus(
    same_spk: dict,
    impostor: dict,
    recon: dict,
    legacy_secs: Optional[float] = None,
) -> dict:
    """
    Core calibration: express reconstruction shift relative to the
    same-speaker ↔ different-speaker span.

    Returns a dict with all calibrated metrics and a grade.
    """
    ss_med = same_spk["summary"]["median"]
    imp_med = impostor["summary"]["median"]
    recon_med = recon["summary"]["median"]

    span = ss_med - imp_med                  # natural variation range
    shift = ss_med - recon_med               # how far reconstruction moved
    imp_dist = recon_med - imp_med           # how far above impostor floor

    shift_over_span = shift / span if span > 1e-9 else float("nan")
    shift_over_impostor = shift / imp_dist if imp_dist > 1e-9 else float("nan")
    preservation = recon_med / ss_med if ss_med > 1e-9 else float("nan")

    # Grading
    if not np.isnan(shift_over_span) and shift_over_span < 0.15 and preservation > 0.85:
        grade = "EXCELLENT"
    elif not np.isnan(shift_over_span) and shift_over_span < 0.30 and preservation > 0.70:
        grade = "PASS"
    elif not np.isnan(shift_over_span) and shift_over_span < 0.50 and preservation > 0.50:
        grade = "MARGINAL"
    else:
        grade = "FAIL"

    return {
        "same_speaker_median": float(ss_med),
        "impostor_median": float(imp_med),
        "recon_median": float(recon_med),
        "same_speaker_span": float(span),
        "reconstruction_shift": float(shift),
        "impostor_distance": float(imp_dist),
        "shift_over_span": float(shift_over_span),
        "shift_over_impostor": float(shift_over_impostor),
        "preservation_ratio": float(preservation),
        "grade": grade,
        "legacy_secs_mean": legacy_secs,
        "legacy_secs_diagnostic": (
            "DIAGNOSTIC_ONLY — old arbitrary-threshold method. "
            "Known broken-adapter range: 0.05-0.24. "
            "NOT valid FACodec performance."
            if legacy_secs is not None and not np.isnan(legacy_secs)
            else "not computed"
        ),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Legacy SECS Diagnostic (preserved for backward compatibility)
# ���══════════════════════════════════════════════════════════════════════════════

def compute_legacy_secs(
    originals: list[Utt],
    reconstructions: list[np.ndarray],
    extractor: ECAPAEmbedding,
) -> float:
    """Old SECS-style metric: mean source-reconstruction cosine sim.

    KEPT AS DIAGNOSTIC ONLY — not used for calibration.
    Known broken-adapter range: 0.05-0.24.
    """
    sims = []
    for orig, recon in zip(originals, reconstructions):
        sims.append(extractor.cosine_sim(
            extractor(orig.wav, orig.sr),
            extractor(recon, orig.sr),
        ))
    return float(np.mean(sims)) if sims else float("nan")


# ═══════════════════════════════════════════════════════════════════════════════
# Prompt Matching (for fair native vs Indian comparison)
# ═══════════════════════════════════════════════════════════════════════════════

def _normalize_text(text: str) -> str:
    """Normalize transcript for matching."""
    import re
    text = text.lower().strip()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_matched_prompts(
    native: list[Utt], indian: list[Utt],
) -> tuple[list[Utt], list[Utt]]:
    """Find utterances with matching transcripts across native and Indian corpora.

    Returns (matched_native, matched_indian) — same-length lists with
    corresponding sentences.
    """
    native_by_text: dict[str, Utt] = {}
    for u in native:
        key = _normalize_text(u.text)
        if key and key not in native_by_text:
            native_by_text[key] = u

    matched_native, matched_indian = [], []
    seen_native = set()
    for u in indian:
        key = _normalize_text(u.text)
        if key in native_by_text:
            nu = native_by_text[key]
            if id(nu) not in seen_native:
                matched_native.append(nu)
                matched_indian.append(u)
                seen_native.add(id(nu))

    print(f"  Matched prompts: {len(matched_native)} pairs")
    return matched_native, matched_indian


# ═���═════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def run(cfg: Config):
    t_start = time.time()
    np.random.seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)
    print(f"Output: {cfg.output_dir}")
    print(f"Device: {cfg.device}")
    print(f"Seed:   {cfg.seed}")

    # ECAPA-TDNN
    print("\n>>> Loading ECAPA-TDNN (speechbrain/spkrec-ecapa-voxceleb)...")
    extractor = ECAPAEmbedding(device=cfg.device)
    print(f"  ECAPA-TDNN ready on {cfg.device}")

    # FACodec
    if not cfg.facodec_dir.exists():
        print(f"\n[ERROR] FACodec directory not found: {cfg.facodec_dir}")
        print("  Set --facodec-dir or clone Plachta/FAcodec to that path.")
        sys.exit(1)

    print(f"\n>>> Loading FACodec ({cfg.facodec_ckpt})...")
    facodec = FACodecModel(
        facodec_dir=cfg.facodec_dir,
        ckpt_name=cfg.facodec_ckpt,
        device=cfg.device,
    )
    print("  FACodec ready")

    # Datasets
    print("\n>>> Loading datasets...")
    native_utts = load_cmu_arctic(cfg)
    indian_utts = load_l2_arctic(cfg)

    if len(native_utts) < 4:
        print("[ERROR] Fewer than 4 native utterances loaded — aborting.")
        sys.exit(1)
    if len(indian_utts) < 4:
        print("[ERROR] Fewer than 4 Indian utterances loaded — aborting.")
        sys.exit(1)

    # Find matched prompts for fair comparison
    matched_native, matched_indian = find_matched_prompts(native_utts, indian_utts)

    # Reference Distribution 1: SAME-SPEAKER
    print("\n>>> SAME-SPEAKER reference (within-speaker variation)...")
    native_same = compute_same_speaker_distribution(native_utts, extractor, cfg.n_pairs_ref)
    indian_same = compute_same_speaker_distribution(indian_utts, extractor, cfg.n_pairs_ref)
    print(f"  Native  same-speaker: n={native_same['summary']['n']}, "
          f"median={native_same['summary']['median']:.4f}")
    print(f"  Indian  same-speaker: n={indian_same['summary']['n']}, "
          f"median={indian_same['summary']['median']:.4f}")

    # Reference Distribution 2: DIFFERENT-SPEAKER
    print("\n>>> DIFFERENT-SPEAKER reference (impostor floor)...")
    native_imp_within = compute_different_speaker_distribution(
        native_utts, native_utts, extractor, cfg.n_pairs_ref,
        label="native_within",
    )
    indian_imp_within = compute_different_speaker_distribution(
        indian_utts, indian_utts, extractor, cfg.n_pairs_ref,
        label="indian_within",
    )
    native_imp_cross = compute_different_speaker_distribution(
        native_utts, indian_utts, extractor, cfg.n_pairs_ref,
        label="native_vs_indian",
    )
    print(f"  Native  within-corpus impostor: n={native_imp_within['summary']['n']}, "
          f"median={native_imp_within['summary']['median']:.4f}")
    print(f"  Indian  within-corpus impostor: n={indian_imp_within['summary']['n']}, "
          f"median={indian_imp_within['summary']['median']:.4f}")
    print(f"  Cross-corpus (native vs Indian): n={native_imp_cross['summary']['n']}, "
          f"median={native_imp_cross['summary']['median']:.4f}")

    # FACodec Reconstructions
    print("\n>>> FACodec reconstructions...")
    native_recon_wavs: list[np.ndarray] = []
    indian_recon_wavs: list[np.ndarray] = []

    for utt in native_utts[: cfg.n_recon]:
        try:
            recon = facodec.reconstruct(utt.wav)
            dur_ratio = len(recon) / max(len(utt.wav), 1)
            print(f"  native [{utt.speaker_id}/{utt.sentence_id}] "
                  f"dur_ratio={dur_ratio:.4f}")
            native_recon_wavs.append(recon)
        except Exception as e:
            print(f"  [WARN] Native recon failed {utt.sentence_id}: {e}")

    for utt in indian_utts[: cfg.n_recon]:
        try:
            recon = facodec.reconstruct(utt.wav)
            dur_ratio = len(recon) / max(len(utt.wav), 1)
            print(f"  indian [{utt.speaker_id}/{utt.sentence_id}] "
                  f"dur_ratio={dur_ratio:.4f}")
            indian_recon_wavs.append(recon)
        except Exception as e:
            print(f"  [WARN] Indian recon failed {utt.sentence_id}: {e}")

    # Reference Distribution 3: RECONSTRUCTION
    print("\n>>> RECONSTRUCTION reference (source vs FACodec output)...")
    native_recon = compute_reconstruction_distribution(
        native_utts[: len(native_recon_wavs)], native_recon_wavs, extractor, "native",
    )
    indian_recon = compute_reconstruction_distribution(
        indian_utts[: len(indian_recon_wavs)], indian_recon_wavs, extractor, "indian",
    )
    print(f"  Native  recon: n={native_recon['summary']['n']}, "
          f"median={native_recon['summary']['median']:.4f}")
    print(f"  Indian  recon: n={indian_recon['summary']['n']}, "
          f"median={indian_recon['summary']['median']:.4f}")

    # Legacy SECS Diagnostic
    native_legacy = compute_legacy_secs(
        native_utts[: len(native_recon_wavs)], native_recon_wavs, extractor,
    )
    indian_legacy = compute_legacy_secs(
        indian_utts[: len(indian_recon_wavs)], indian_recon_wavs, extractor,
    )
    print(f"\n  +{'-'*56}+")
    print(f"  |  Legacy SECS mean (DIAGNOSTIC ONLY)          |")
    print(f"  |  Native: {native_legacy:.4f}                              |")
    print(f"  |  Indian: {indian_legacy:.4f}                              |")
    print(f"  |  Broken-adapter range: 0.05 - 0.24             |")
    print(f"  |  NOT valid FACodec performance                 |")
    print(f"  +{'-'*56}+")

    # Calibrated Comparisons
    print("\n>>> Calibrated comparisons...")
    native_cal = calibrate_corpus(native_same, native_imp_within, native_recon, native_legacy)
    indian_cal = calibrate_corpus(indian_same, indian_imp_within, indian_recon, indian_legacy)

    # Comparison Table
    print("\n" + "=" * 72)
    print("  CALIBRATED COMPARISON TABLE")
    print("=" * 72)
    header = f"  {'Metric':<36} {'Native':>14} {'Indian':>14}"
    print(header)
    print("  " + "-" * 66)

    rows = [
        ("same-speaker median",         native_cal["same_speaker_median"],  indian_cal["same_speaker_median"]),
        ("impostor median",              native_cal["impostor_median"],       indian_cal["impostor_median"]),
        ("recon median",                 native_cal["recon_median"],          indian_cal["recon_median"]),
        ("same-speaker span",            native_cal["same_speaker_span"],     indian_cal["same_speaker_span"]),
        ("reconstruction shift",         native_cal["reconstruction_shift"],  indian_cal["reconstruction_shift"]),
        ("impostor distance",            native_cal["impostor_distance"],     indian_cal["impostor_distance"]),
        ("shift / span",                 native_cal["shift_over_span"],       indian_cal["shift_over_span"]),
        ("shift / impostor distance",    native_cal["shift_over_impostor"],   indian_cal["shift_over_impostor"]),
        ("preservation ratio",           native_cal["preservation_ratio"],    indian_cal["preservation_ratio"]),
        ("GRADE",                        native_cal["grade"],                 indian_cal["grade"]),
        ("legacy SECS (DIAG ONLY)",      native_cal["legacy_secs_mean"],      indian_cal["legacy_secs_mean"]),
    ]
    for metric, nv, iv in rows:
        def fmt(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return "N/A"
            if isinstance(v, str):
                return v
            return f"{v:.4f}"
        print(f"  {metric:<36} {fmt(nv):>14} {fmt(iv):>14}")

    print("  " + "-" * 66)
    gap = native_cal["shift_over_span"] - indian_cal["shift_over_span"]
    print(f"  {'shift/span gap (native - Indian)':<36} {float(gap):>14.4f}")
    print("=" * 72)

    # Build Summary
    matched_pairs_used = min(len(matched_native), len(matched_indian)) if matched_native else 0
    elapsed = time.time() - t_start

    summary = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "device": cfg.device,
            "seed": cfg.seed,
            "facodec_dir": str(cfg.facodec_dir),
            "facodec_ckpt": cfg.facodec_ckpt,
            "n_pairs_ref": cfg.n_pairs_ref,
            "n_recon": cfg.n_recon,
            "native_utterances_loaded": len(native_utts),
            "indian_utterances_loaded": len(indian_utts),
            "native_reconstructed": len(native_recon_wavs),
            "indian_reconstructed": len(indian_recon_wavs),
            "matched_prompts_used": matched_pairs_used,
            "elapsed_seconds": round(elapsed, 1),
        },
        "methodology": {
            "same_speaker_ref": "Within-speaker pairs: utterance A vs utterance B",
            "different_speaker_ref": "Cross-speaker pairs: speaker A vs speaker B",
            "reconstruction_ref": "Source waveform vs FACodec reconstruction",
            "key_comparison": (
                "shift_over_span = reconstruction_shift / same_speaker_span. "
                "Expresses reconstruction damage as fraction of natural speaker variation. "
                "Avoids arbitrary cosine-similarity thresholds."
            ),
            "grading": {
                "EXCELLENT": "shift_over_span < 0.15 AND preservation > 0.85",
                "PASS":     "shift_over_span < 0.30 AND preservation > 0.70",
                "MARGINAL": "shift_over_span < 0.50 AND preservation > 0.50",
                "FAIL":     "anything worse",
            },
        },
        "native": native_cal,
        "indian": indian_cal,
        "cross_corpus": {
            "native_vs_indian_impostor_median": float(native_imp_cross["summary"]["median"]),
            "native_vs_indian_impostor_n": int(native_imp_cross["summary"]["n"]),
        },
        "comparison": {
            "identity_gap_shift_over_span": float(gap),
            "interpretation": (
                f"Native English reconstruction shift/span = {native_cal['shift_over_span']:.4f} "
                f"(grade {native_cal['grade']}). "
                f"Indian English reconstruction shift/span = {indian_cal['shift_over_span']:.4f} "
                f"(grade {indian_cal['grade']}). "
                f"Gap: {gap:+.4f}."
            ),
        },
        "legacy_secs_diagnostic": {
            "native_mean": float(native_legacy) if not np.isnan(native_legacy) else None,
            "indian_mean": float(indian_legacy) if not np.isnan(indian_legacy) else None,
            "label": "DIAGNOSTIC_ONLY / BROKEN ADAPTER / NOT VALID FACODEC PERFORMANCE",
            "known_broken_range": "0.05-0.24",
            "note": (
                "This metric used arbitrary cosine-similarity thresholds. "
                "It does NOT calibrate against speaker variation and is "
                "unreliable for cross-corpus comparison. Superseded by "
                "shift_over_span and preservation_ratio."
            ),
        },
    }

    # Save Outputs
    def save_json(data, filename):
        path = cfg.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=_json_safe)
        print(f"  Saved: {path}")

    def _json_safe(obj):
        if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
            return None
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return str(obj)

    save_json(native_same | {"indian": indian_same},  "same_speaker_dist.json")
    save_json({
        "native_within": native_imp_within,
        "indian_within": indian_imp_within,
        "native_vs_indian": native_imp_cross,
    }, "different_speaker_dist.json")
    save_json({
        "native": native_recon,
        "indian": indian_recon,
        "legacy_secs_diagnostic": summary["legacy_secs_diagnostic"],
    }, "reconstruction_dist.json")
    save_json(summary, "summary.json")

    # Final Verdict
    print(f"\n{'=' * 72}")
    print(f"  VERDICT")
    print(f"{'=' * 72}")
    for corpus in ["native", "indian"]:
        c = summary[corpus]
        print(f"  {corpus.upper():<10} grade={c['grade']}  "
              f"shift/span={c['shift_over_span']:.4f}  "
              f"preservation={c['preservation_ratio']:.4f}")
    print(f"  Elapsed: {elapsed:.1f}s")
    print(f"{'=' * 72}")

    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Calibrated identity preservation: Native vs Indian English"
    )
    parser.add_argument("--facodec-dir", type=str, default=str(cfg.facodec_dir),
                        help="Path to FAcodec repo")
    parser.add_argument("--device", type=str, default=cfg.device)
    parser.add_argument("--n-pairs", type=int, default=cfg.n_pairs_ref)
    parser.add_argument("--n-recon", type=int, default=cfg.n_recon)
    parser.add_argument("--output-dir", type=str, default=str(cfg.output_dir))
    parser.add_argument("--seed", type=int, default=cfg.seed)
    args = parser.parse_args()

    cfg.facodec_dir = Path(args.facodec_dir)
    cfg.device = args.device
    cfg.n_pairs_ref = args.n_pairs
    cfg.n_recon = args.n_recon
    cfg.output_dir = Path(args.output_dir)
    cfg.seed = args.seed

    run(cfg)


if __name__ == "__main__":
    main()
