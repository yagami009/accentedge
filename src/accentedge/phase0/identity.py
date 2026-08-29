"""
Speaker identity encoders for Phase 0 evaluation.

Provides a unified interface to multiple speaker encoders (ECAPA-TDNN,
WavLM-based verifier, plus a Resemblyzer-style encoder).  Computes cosine
similarity and distance between audio segments to verify that accent
transformation preserves speaker identity.

Dependencies (optional, loaded lazily):
- speechbrain >= 1.0 — ECAPA-TDNN encoder
- transformers >= 4.30 — WavLM model for speaker verification
- torch >= 2.0 — required by the above
- resemblyzer — optional third encoder (can be absent)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Data classes ───────────────────────────────────────────────────────────

@dataclass
class SpeakerEncoderResult:
    """Aggregated speaker-encoder comparison result."""
    encoder_names: list[str]
    distances: list[float]
    similarities: list[float]
    mean_distance: float
    mean_similarity: float
    individual_results: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "encoder_names": self.encoder_names,
            "distances": self.distances,
            "similarities": self.similarities,
            "mean_distance": self.mean_distance,
            "mean_similarity": self.mean_similarity,
            "individual_results": self.individual_results,
        }


# ── Embedding helpers (reused from probes) ─────────────────────────────────

def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    sim = float(np.dot(a, b) / (na * nb))
    return float(np.clip(sim, -1.0, 1.0))


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    return 1.0 - _cosine_similarity(a, b)


def _try_load_ecapa():
    """Lazy-load ECAPA-TDNN from SpeechBrain. Returns None if unavailable."""
    try:
        import torch  # noqa: F401
        from speechbrain.inference import EncoderClassifier  # type: ignore
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_ecapa",
        )
        return model
    except ImportError:
        return None
    except Exception as exc:
        logger.warning(f"ECAPA-TDNN load failed: {exc}")
        return None


def _try_load_wavlm():
    """Lazy-load WavLM for speaker verification. Returns None if unavailable."""
    try:
        import torch  # noqa: F401
        from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor  # type: ignore
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        return model, extractor
    except ImportError:
        return None
    except Exception as exc:
        logger.warning(f"WavLM load failed: {exc}")
        return None


def _try_load_resemblyzer():
    """Lazy-load Resemblyzer-style encoder. Returns None if unavailable."""
    try:
        from resemblyzer import preprocess_wav, VoiceEncoder  # type: ignore
        encoder = VoiceEncoder()
        return encoder, preprocess_wav
    except ImportError:
        logger.debug("resemblyzer not installed — skipping")
        return None
    except Exception as exc:
        logger.warning(f"Resemblyzer load failed: {exc}")
        return None


def _embed_ecapa(model, waveform: np.ndarray, sr: int) -> np.ndarray:
    import torch
    wf = torch.from_numpy(waveform).float().unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(wf).squeeze().cpu().numpy()
    return emb.astype(np.float32)


def _embed_wavlm(model_tuple, waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract embedding using WavLM. WavLM expects 16kHz input."""
    import librosa  # type: ignore
    import torch
    model, extractor = model_tuple
    # Resample to 16kHz if needed
    if sr != 16000:
        wf = librosa.resample(waveform.astype(np.float32), orig_sr=sr, target_sr=16000)
    else:
        wf = waveform.astype(np.float32)
    inputs = extractor(wf, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    hidden = outputs.hidden_states[-1]
    emb = hidden.mean(dim=1).squeeze().cpu().numpy()
    return emb.astype(np.float32)


def _embed_resemblyzer(model_tuple, waveform: np.ndarray, sr: int) -> np.ndarray:
    encoder, preprocess_wav = model_tuple
    # Resemblyzer expects 16kHz mono float32
    import librosa  # type: ignore
    wf_16k = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
    wf_16k = wf_16k.astype(np.float32)
    if wf_16k.ndim > 1:
        wf_16k = wf_16k.mean(axis=1)
    emb = encoder.embed_utterance(wf_16k)
    return emb.astype(np.float32)


def _mfcc_embedding(waveform: np.ndarray, sr: int, n_mfcc: int = 40) -> np.ndarray:
    """Fallback MFCC embedding when ML models are unavailable."""
    import librosa  # type: ignore
    wf = np.asarray(waveform, dtype=np.float32)
    if wf.ndim > 1:
        wf = wf.mean(axis=1)
    min_samples = int(0.1 * sr)
    if len(wf) < min_samples:
        wf = np.pad(wf, (0, min_samples - len(wf)), mode="constant")
    mfcc = librosa.feature.mfcc(y=wf, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=128)
    return mfcc.mean(axis=1).astype(np.float32)


# ── SpeakerEncoder ────────────────────────────────────────────────────────

class SpeakerEncoder:
    """
    Unified interface to multiple speaker encoders.

    Automatically discovers available encoders on initialization.
    Each encoder has a name and an embed function.

    Available encoder slots:
      - ecapa  : SpeechBrain ECAPA-TDNN
      - wavlm  : Microsoft WavLM base-plus-sv
      - resemblyzer : Resemblyzer VoiceEncoder
      - mfcc   : librosa MFCC fallback (always available)
    """

    ENCODER_NAMES = ["ecapa", "wavlm", "resemblyzer", "mfcc"]

    def __init__(self, sample_rate: int = 22050, encoders=None):
        self.sample_rate = sample_rate
        self._encoders: dict[str, dict] = {}

        # ECAPA-TDNN
        ecapa = _try_load_ecapa()
        if ecapa is not None:
            self._encoders["ecapa"] = {
                "model": ecapa,
                "embed": lambda wf, sr: _embed_ecapa(ecapa, wf, sr),
            }

        # WavLM
        wavlm = _try_load_wavlm()
        if wavlm is not None:
            self._encoders["wavlm"] = {
                "model": wavlm,
                "embed": lambda wf, sr: _embed_wavlm(wavlm, wf, sr),
            }

        # Resemblyzer
        resemblyzer = _try_load_resemblyzer()
        if resemblyzer is not None:
            self._encoders["resemblyzer"] = {
                "model": resemblyzer,
                "embed": lambda wf, sr: _embed_resemblyzer(resemblyzer, wf, sr),
            }

        # MFCC fallback — always available if librosa is installed
        try:
            import librosa  # noqa: F401
            self._encoders["mfcc"] = {
                "model": None,
                "embed": lambda wf, sr: _mfcc_embedding(wf, sr),
            }
        except ImportError:
            pass

        if not self._encoders:
            raise ImportError(
                "No speaker encoder backends available. "
                "Install at least one of: speechbrain, transformers+torch, "
                "resemblyzer, librosa."
            )

        logger.info(
            f"SpeakerEncoder initialized with: {list(self._encoders.keys())}"
        )

    @property
    def available_encoders(self) -> list[str]:
        return list(self._encoders.keys())

    def compute_embedding(self, waveform: np.ndarray, encoder_name: str) -> np.ndarray:
        """Extract embedding using a specific encoder."""
        if encoder_name not in self._encoders:
            raise ValueError(
                f"Unknown encoder '{encoder_name}'. "
                f"Available: {self.available_encoders}"
            )
        wf = np.asarray(waveform, dtype=np.float32)
        if wf.ndim > 1:
            wf = wf.mean(axis=1)
        return self._encoders[encoder_name]["embed"](wf, self.sample_rate)

    def compute_similarity(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Mean cosine similarity across all active encoders."""
        result = self.compare(audio1, audio2)
        if not result.encoder_names:
            raise RuntimeError("No encoders available for similarity computation")
        return result.mean_similarity

    def compute_distance(self, audio1: np.ndarray, audio2: np.ndarray) -> float:
        """Mean cosine distance across all active encoders."""
        result = self.compare(audio1, audio2)
        if not result.encoder_names:
            raise RuntimeError("No encoders available for distance computation")
        return result.mean_distance

    def compare(self, audio1: np.ndarray, audio2: np.ndarray) -> SpeakerEncoderResult:
        """Run all encoders and return aggregated distances and similarities."""
        return self._compute_all(audio1, audio2)

    def _compute_all(self, audio1: np.ndarray, audio2: np.ndarray) -> SpeakerEncoderResult:
        enc_names = []
        dists = []
        sims = []
        individual = {}

        for name in self._encoders:
            try:
                emb1 = self.compute_embedding(audio1, name)
                emb2 = self.compute_embedding(audio2, name)
                dist = 1.0 - _cosine_similarity(emb1, emb2)
                sim = _cosine_similarity(emb1, emb2)
                dists.append(dist)
                sims.append(sim)
                enc_names.append(name)
                individual[name] = {
                    "distance": float(dist),
                    "similarity": float(sim),
                }
            except Exception as exc:
                logger.warning(f"Encoder '{name}' failed: {exc}")

        if not dists:
            return SpeakerEncoderResult(
                encoder_names=[], distances=[], similarities=[],
                mean_distance=1.0, mean_similarity=0.0,
            )

        return SpeakerEncoderResult(
            encoder_names=enc_names,
            distances=dists,
            similarities=sims,
            mean_distance=float(np.mean(dists)),
            mean_similarity=float(np.mean(sims)),
            individual_results=individual,
        )


# ── Standalone wrappers ──────────────────────────────────────────────────────

def compute_similarity(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sample_rate: int = 22050,
    encoders: Optional[list[str]] = None,
) -> float:
    """Compute mean cosine similarity across available speaker encoders."""
    se = SpeakerEncoder(sample_rate=sample_rate, encoders=encoders)
    return se.compute_similarity(audio1, audio2)


def compute_distance(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sample_rate: int = 22050,
    encoders: Optional[list[str]] = None,
) -> float:
    """Compute mean cosine distance across available speaker encoders."""
    se = SpeakerEncoder(sample_rate=sample_rate, encoders=encoders)
    return se.compute_distance(audio1, audio2)


def compute_speaker_distance(
    audio1: np.ndarray,
    audio2: np.ndarray,
    sample_rate: int = 22050,
    encoders: Optional[list[str]] = None,
) -> SpeakerEncoderResult:
    """Run all encoders and return the aggregated result."""
    se = SpeakerEncoder(sample_rate=sample_rate, encoders=encoders)
    return se.compare(audio1, audio2)


def _cosine_distance(a, b):
    """L2-normalized cosine distance."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 1.0
    return float(1.0 - np.dot(a, b) / (na * nb))


def _cosine_similarity(a, b):
    """L2-normalized cosine similarity."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))



