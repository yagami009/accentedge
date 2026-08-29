"""
Accent pronunciation probes for Gate -1B measurement validity.

Each probe targets a specific pronunciation dimension (RHO, FLAP, TH, ASP,
RET, VW, RED, STR) and classifies an audio segment as target-like (US-neutral)
or substitute-like (Indian-English realization) using self-supervised embeddings.

Dependencies (optional, loaded lazily):
- speechbrain >= 1.0  — provides ECAPA-TDNN speaker/encoding model
- transformers >= 4.30 — provides WavLM for verification
- torch >= 2.0       — required by the above

If a dependency is missing at use-time, the calling function receives a clear
ImportError rather than a crash at import time.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ── Feature inventory ──────────────────────────────────────────────────────

class ProbeDimension(Enum):
    """Pronunciation dimensions targeted by Gate -1B probes."""
    RHO = "RHO"          # Rhoticity: post-vocalic /r/ realization
    FLAP = "FLAP"        # Intervocalic /t/ flapping (US) vs. crisp stop (IN)
    TH = "TH"            # /θ ð/ dental fricatives vs. /t d dʰ/
    ASP = "ASP"          # Stop aspiration: /p t k/ aspiration strength
    RET = "RET"          # Retroflex vs. alveolar /t d n/
    VW = "VW"            # /v/–/w/ labiodental vs. labio-velar approximant
    RED = "RED"          # Vowel reduction quality in unstressed syllables
    STR = "STR"          # Lexical stress pattern differences


DIMENSION_DESCRIPTIONS = {
    ProbeDimension.RHO:  "Post-vocalic rhoticity",
    ProbeDimension.FLAP: "Intervocalic /t/ flapping vs. crisp stop",
    ProbeDimension.TH:   "Dental fricative /θ ð/ realization",
    ProbeDimension.ASP:  "Stop aspiration strength",
    ProbeDimension.RET:  "Retroflex vs. alveolar coronal stops",
    ProbeDimension.VW:   "/v/ vs. /w/ labial approximant",
    ProbeDimension.RED:  "Vowel reduction in unstressed position",
    ProbeDimension.STR:  "Lexical stress pattern",
}


# ── Data classes ───────────────────────────────────��───────────────────────

@dataclass
class ProbeResult:
    """Result of running a single accent probe on an audio segment."""
    dimension: ProbeDimension
    phone_label: str
    classification: str          # "target" or "substitute"
    confidence: float            # 0..1, higher = more certain
    distance_to_native: float    # cosine distance to native-US centroid
    distance_to_indian: float    # cosine distance to Indian-English centroid
    embedding: Optional[np.ndarray] = None  # raw embedding vector

    def to_dict(self) -> dict:
        return {
            "dimension": self.dimension.value,
            "phone_label": self.phone_label,
            "classification": self.classification,
            "confidence": self.confidence,
            "distance_to_native": self.distance_to_native,
            "distance_to_indian": self.distance_to_indian,
            "has_embedding": self.embedding is not None,
        }


@dataclass
class ProbeValidationResult:
    """Result of a probe-validity test (Gate -1B)."""
    probe_passes: bool
    us_tokens_correct: int
    us_tokens_total: int
    indian_deviant_correct: int
    indian_deviant_total: int
    indian_already_target_correct: int
    indian_already_target_total: int
    overall_accuracy: float
    details: list[ProbeResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "probe_passes": self.probe_passes,
            "us_tokens_correct": self.us_tokens_correct,
            "us_tokens_total": self.us_tokens_total,
            "indian_deviant_correct": self.indian_deviant_correct,
            "indian_deviant_total": self.indian_deviant_total,
            "indian_already_target_correct": self.indian_already_target_correct,
            "indian_already_target_total": self.indian_already_target_total,
            "overall_accuracy": self.overall_accuracy,
            "detail_count": len(self.details),
        }


# ── Feature extraction ────────────────────────────────────────────────────

def _extract_mfcc_like(waveform: np.ndarray, sr: int = 22050,
                       n_mfcc: int = 40) -> np.ndarray:
    """
    Compute a compact MFCC-based representation as a fallback when
    self-supervised models are unavailable.

    This is intentionally lightweight — it exists so probe validation tests
    can run without torch/speechbrain installed.  The centroid-based
    classifier works with any fixed-length feature vector.
    """
    try:
        import librosa  # type: ignore
    except ImportError:
        raise ImportError(
            "librosa is required for fallback feature extraction. "
            "Install it with: pip install librosa"
        )

    # Ensure mono float32
    wf = np.asarray(waveform, dtype=np.float32)
    if wf.ndim > 1:
        wf = wf.mean(axis=1)

    # Pad short clips to at least 0.1s
    min_samples = int(0.1 * sr)
    if len(wf) < min_samples:
        wf = np.pad(wf, (0, min_samples - len(wf)), mode="constant")

    mfcc = librosa.feature.mfcc(
        y=wf, sr=sr, n_mfcc=n_mfcc, n_fft=512, hop_length=128
    )
    # Mean-pool over time → [n_mfcc] vector
    embedding = mfcc.mean(axis=1).astype(np.float32)
    return embedding


def _try_load_ecapa_tdnn():
    """Try to load a SpeechBrain ECAPA-TDNN encoder. Returns None if unavailable."""
    try:
        import torch  # noqa: F401
        from speechbrain.inference import EncoderClassifier  # type: ignore
        model = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/speechbrain_ecapa",
        )
        return model
    except ImportError:
        logger.debug("speechbrain/torch not available — using MFCC fallback")
        return None
    except Exception as exc:
        logger.warning(f"Failed to load ECAPA-TDNN: {exc} — using MFCC fallback")
        return None


def _try_load_wavlm():
    """Try to load a WavLM-based verifier. Returns None if unavailable."""
    try:
        import torch  # noqa: F401
        from transformers import WavLMForXVector, Wav2Vec2FeatureExtractor  # type: ignore
        model = WavLMForXVector.from_pretrained("microsoft/wavlm-base-plus-sv")
        extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "microsoft/wavlm-base-plus-sv"
        )
        return model, extractor
    except ImportError:
        logger.debug("transformers/torch not available — using MFCC fallback")
        return None
    except Exception as exc:
        logger.warning(f"Failed to load WavLM: {exc} — using MFCC fallback")
        return None


# ── Embedding extraction ──────────────────────────────────────────────────

def _embed_ecapa(model, waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract embedding using SpeechBrain ECAPA-TDNN."""
    import torch
    wf = torch.from_numpy(waveform).float().unsqueeze(0)
    with torch.no_grad():
        emb = model.encode_batch(wf).squeeze().cpu().numpy()
    return emb.astype(np.float32)


def _embed_wavlm(model_tuple, waveform: np.ndarray, sr: int) -> np.ndarray:
    """Extract embedding using WavLM. WavLM expects 16kHz input."""
    try:
        import librosa  # type: ignore
    except ImportError:
        raise ImportError("librosa is required for WavLM resampling")
    import torch
    model, extractor = model_tuple
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


def extract_embedding(
    waveform: np.ndarray,
    sr: int = 22050,
    prefer: str = "ecapa",
) -> np.ndarray:
    """
    Extract a speaker/pronunciation embedding from a waveform.

    Preference order: ECAPA-TDNN → WavLM → MFCC fallback.

    Returns a fixed-length numpy vector.
    """
    wf = np.asarray(waveform, dtype=np.float32)
    if wf.ndim > 1:
        wf = wf.mean(axis=1)

    if prefer == "ecapa":
        model = _try_load_ecapa_tdnn()
        if model is not None:
            try:
                return _embed_ecapa(model, wf, sr)
            except Exception:
                pass  # fall through to next option
    elif prefer == "wavlm":
        model_tuple = _try_load_wavlm()
        if model_tuple is not None:
            try:
                return _embed_wavlm(model_tuple, wf, sr)
            except Exception:
                pass  # fall through to MFCC

    # Try WavLM if ECAPA wasn't preferred or failed
    if prefer != "wavlm":
        model_tuple = _try_load_wavlm()
        if model_tuple is not None:
            try:
                return _embed_wavlm(model_tuple, wf, sr)
            except Exception:
                pass  # fall through to MFCC

    # Fallback
    return _extract_mfcc_like(wf, sr)


# ── AccentProbe ────────────────────────────────────────────────────────────

class AccentProbe:
    """
    Accent pronunciation probe.

    Given an audio segment and its phone label, extracts an embedding and
    classifies whether the realization matches the target (US-neutral) or
    substitute (Indian-English) using pre-built reference centroids.

    The probe uses cosine distance to native-US and Indian-English reference
    centroids for the given phone.  Classification is whichever centroid is
    closer; confidence is the margin between distances.
    """

    def __init__(
        self,
        dimension: ProbeDimension,
        reference_dir: Optional[Path] = None,
        embedding_fn: Optional[callable] = None,
        sample_rate: int = 22050,
    ):
        self.dimension = dimension
        self.sample_rate = sample_rate
        self.embedding_fn = embedding_fn or extract_embedding

        # {phone_label: native_centroid} and {phone_label: indian_centroid}
        self._native_centroids: dict[str, np.ndarray] = {}
        self._indian_centroids: dict[str, np.ndarray] = {}

        if reference_dir is not None:
            self.load_centroids(reference_dir)

    # ── centroid management ──────────────────────────────────────────────

    def load_centroids(self, reference_dir: Path) -> None:
        """
        Load pre-computed reference centroids from a directory.

        Expected layout:
            reference_dir/
                native-US/
                    RHO/  (flattened .npy files, one per phone)
                    FLAP/
                    ...
                Indian-English/
                    RHO/
                    FLAP/
                    ...

        Each .npy file is a [n_examples, embedding_dim] array; the centroid
        is the mean over axis 0.
        """
        ref = Path(reference_dir)
        native_root = ref / "native-US"
        indian_root = ref / "Indian-English"

        dim_label = self.dimension.value

        for root, label in [(native_root, "native"), (indian_root, "indian")]:
            dim_dir = root / dim_label
            if not dim_dir.is_dir():
                logger.warning(f"No reference directory: {dim_dir}")
                continue
            centroids: dict[str, np.ndarray] = {}
            for npy_path in sorted(dim_dir.glob("*.npy")):
                phone = npy_path.stem
                arr = np.load(npy_path)
                if arr.ndim == 1:
                    centroid = arr
                else:
                    centroid = arr.mean(axis=0)
                centroids[phone] = centroid.astype(np.float32)
            if label == "native":
                self._native_centroids = centroids
            else:
                self._indian_centroids = centroids

        logger.debug(
            f"Loaded centroids for {self.dimension.value}: "
            f"{len(self._native_centroids)} native, "
            f"{len(self._indian_centroids)} indian"
        )

    def set_centroids(
        self,
        native: dict[str, np.ndarray],
        indian: dict[str, np.ndarray],
    ) -> None:
        """Directly set centroid dictionaries (used in tests / build step)."""
        self._native_centroids = {k: v.astype(np.float32) for k, v in native.items()}
        self._indian_centroids = {k: v.astype(np.float32) for k, v in indian.items()}

    # ── classification ───────────────────────────────────────────────────

    def classify(self, waveform: np.ndarray, phone_label: str) -> ProbeResult:
        """
        Classify a phone segment as target-like or substitute-like.

        Args:
            waveform: Audio segment (float32 numpy array).
            phone_label: IPA or ARPABET phone label, e.g. "r", "ɾ", "t".

        Returns:
            ProbeResult with classification, confidence, and distances.
        """
        if not self._native_centroids or not self._indian_centroids:
            raise RuntimeError(
                f"No reference centroids loaded for probe {self.dimension.value}. "
                "Call load_centroids() or set_centroids() first."
            )

        embedding = self.embedding_fn(waveform, self.sample_rate)

        # Find nearest centroids for the given phone label
        native_cent = self._find_nearest_centroid(embedding, self._native_centroids)
        indian_cent = self._find_nearest_centroid(embedding, self._indian_centroids)

        dist_native = _cosine_distance(embedding, native_cent)
        dist_indian = _cosine_distance(embedding, indian_cent)

        if dist_native <= dist_indian:
            classification = "target"
            confidence = float(np.clip(1.0 - dist_native, 0.0, 1.0))
        else:
            classification = "substitute"
            confidence = float(np.clip(1.0 - dist_indian, 0.0, 1.0))

        return ProbeResult(
            dimension=self.dimension,
            phone_label=phone_label,
            classification=classification,
            confidence=confidence,
            distance_to_native=float(dist_native),
            distance_to_indian=float(dist_indian),
            embedding=embedding.copy(),
        )

    @staticmethod
    def _find_nearest_centroid(
        embedding: np.ndarray, centroids: dict[str, np.ndarray]
    ) -> np.ndarray:
        """Return the centroid with smallest cosine distance to embedding."""
        best_phone = None
        best_dist = float("inf")
        for phone, cent in centroids.items():
            d = _cosine_distance(embedding, cent)
            if d < best_dist:
                best_dist = d
                best_phone = phone
        if best_phone is None:
            raise ValueError("Empty centroid dictionary")
        return centroids[best_phone]


# ── Reference centroid builder ─────────────────────────────────────────────

def build_reference_centroids(
    corpus_dir: Path,
    output_dir: Path,
    embedding_fn: Optional[callable] = None,
    sample_rate: int = 22050,
) -> dict[str, dict[str, dict[str, np.ndarray]]]:
    """
    Compute mean embeddings from a reference corpus organized by
    speaker_accent/phone.

    Args:
        corpus_dir: Root of the reference corpus.
            Layout: corpus_dir/<speaker_accent>/<phone_label>/*.wav
        output_dir: Where to save per-phone .npy centroid files.
        embedding_fn: Function to extract embeddings. Uses default if None.
        sample_rate: Expected sample rate of audio files.

    Returns:
        Nested dict: {accent: {phone: centroid_array}}
    """
    import soundfile as sf  # local import, always available

    corpus_dir = Path(corpus_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if embedding_fn is None:
        embedding_fn = extract_embedding

    results: dict[str, dict[str, dict[str, np.ndarray]]] = {}

    for accent_dir in sorted(corpus_dir.iterdir()):
        if not accent_dir.is_dir():
            continue
        accent = accent_dir.name
        results[accent] = {}

        for dim_dir in sorted(accent_dir.iterdir()):
            if not dim_dir.is_dir():
                continue
            dim_label = dim_dir.name  # e.g. "RHO"
            accent_dim_out = output_dir / accent / dim_label
            accent_dim_out.mkdir(parents=True, exist_ok=True)
            phone_embeddings: dict[str, list[np.ndarray]] = {}

            for wav_path in sorted(dim_dir.glob("*.wav")):
                phone_label = wav_path.stem  # filename stem = phone label
                try:
                    wf, sr_read = sf.read(str(wav_path), dtype=np.float32)
                    if isinstance(sr_read, int) and sr_read != sample_rate:
                        continue
                    if hasattr(sr_read, 'samplerate') and sr_read.samplerate != sample_rate:
                        continue
                    emb = embedding_fn(wf, sample_rate)
                    if phone_label not in phone_embeddings:
                        phone_embeddings[phone_label] = []
                    phone_embeddings[phone_label].append(emb)
                except Exception as exc:
                    logger.warning(f"Skipping {wav_path}: {exc}")

            if dim_label not in results[accent]:
                results[accent][dim_label] = {}

            for phone_label, embs in phone_embeddings.items():
                if not embs:
                    continue
                stacked = np.stack(embs, axis=0)
                centroid = stacked.mean(axis=0).astype(np.float32)
                npy_path = accent_dim_out / f"{phone_label}.npy"
                np.save(str(npy_path), centroid)
                results[accent][dim_label][phone_label] = centroid

    return results


# ── Probe validation ───────────────────────────────────────────────────────

def validate_probe(
    probe: AccentProbe,
    us_tokens: list[tuple[np.ndarray, str]],
    indian_deviant_tokens: list[tuple[np.ndarray, str]],
    indian_target_tokens: list[tuple[np.ndarray, str]],
) -> ProbeValidationResult:
    """
    Test that a probe correctly classifies three categories of tokens:

    1. Known natural US tokens → should classify as "target"
    2. Known deviant Indian tokens → should classify as "substitute"
    3. Already-target Indian tokens → should classify as "target"

    This is the Gate -1B probe-validity test.

    Args:
        probe: AccentProbe instance with loaded centroids.
        us_tokens: list of (waveform, phone_label) — known US realizations.
        indian_deviant_tokens: list of (waveform, phone_label) — known deviant IN.
        indian_target_tokens: list of (waveform, phone_label) — IN already target.

    Returns:
        ProbeValidationResult with pass/fail and per-token details.
    """
    details: list[ProbeResult] = []
    us_correct = 0
    indian_dev_correct = 0
    indian_tgt_correct = 0

    for wf, phone in us_tokens:
        result = probe.classify(wf, phone)
        details.append(result)
        if result.classification == "target":
            us_correct += 1

    for wf, phone in indian_deviant_tokens:
        result = probe.classify(wf, phone)
        details.append(result)
        if result.classification == "substitute":
            indian_dev_correct += 1

    for wf, phone in indian_target_tokens:
        result = probe.classify(wf, phone)
        details.append(result)
        if result.classification == "target":
            indian_tgt_correct += 1

    n_us = len(us_tokens)
    n_ind_dev = len(indian_deviant_tokens)
    n_ind_tgt = len(indian_target_tokens)
    total = n_us + n_ind_dev + n_ind_tgt
    correct = us_correct + indian_dev_correct + indian_tgt_correct

    accuracy = correct / total if total > 0 else 0.0
    passes = accuracy >= 0.8  # require ≥80% accuracy for probe validity

    return ProbeValidationResult(
        probe_passes=passes,
        us_tokens_correct=us_correct,
        us_tokens_total=n_us,
        indian_deviant_correct=indian_dev_correct,
        indian_deviant_total=n_ind_dev,
        indian_already_target_correct=indian_tgt_correct,
        indian_already_target_total=n_ind_tgt,
        overall_accuracy=accuracy,
        details=details,
    )


# ── Utilities ──────────────────────────────────────────────────────────────

def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance between two vectors (1 - cosine similarity)."""
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 1.0
    sim = float(np.dot(a, b) / (na * nb))
    sim = float(np.clip(sim, -1.0, 1.0))
    return 1.0 - sim

