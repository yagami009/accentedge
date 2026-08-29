"""
Identity/timbre transfer implementations for Phase 0.

Provides pluggable identity transfer methods abstracted behind the
IdentityTransfer interface. Each implementation tracks whether accent
information leaked through during the transfer process.

Available implementations:
- SimpleVoiceConversionTransfer: spectral envelope matching
- SpeakerEmbeddingTransfer: speaker-embedding conditioning with
  optional Seed-VC backend and spectral fallback
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

try:
    import librosa

    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    logger.warning("librosa not available — spectral identity transfer will use fallbacks")

try:
    import soundfile as sf

    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False


@dataclass
class TransferResult:
    """Result of an identity transfer operation."""

    waveform: np.ndarray
    sample_rate: int
    accent_leaked: bool = False
    leak_confidence: float = 0.0
    modifications: list = field(default_factory=list)
    source_features: dict = field(default_factory=dict)
    target_features: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "accent_leaked": self.accent_leaked,
            "leak_confidence": self.leak_confidence,
            "modifications": self.modifications,
            "source_features": self.source_features,
            "target_features": self.target_features,
        }


class IdentityTransfer(ABC):
    """Abstract interface for identity/timbre transfer.

    Implementations transfer speaker identity from source audio
    to a target audio while tracking accent leakage.
    """

    @abstractmethod
    def transfer(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        target_audio: np.ndarray,
        target_sr: int,
        strength: float = 1.0,
    ) -> TransferResult:
        """Transfer source speaker identity to target audio.

        Args:
            source_audio: Source speaker waveform
            source_sr: Source sample rate
            target_audio: Target audio to transfer identity onto
            target_sr: Target sample rate
            strength: Blending strength [0, 1]

        Returns:
            TransferResult with output waveform and metadata
        """
        pass

    @abstractmethod
    def extract_speaker_embedding(
        self, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        """Extract a speaker embedding from audio.

        Returns a vector representing speaker characteristics.
        """
        pass

    def detect_accent_leak(
        self,
        source_audio: np.ndarray,
        output_audio: np.ndarray,
        sr: int,
        threshold: float = 0.3,
    ) -> tuple:
        """Detect whether source accent information leaked through.

        Compares spectral features of source and output. If they are
        too similar, accent characteristics from the source were preserved.

        Returns:
            (leaked: bool, confidence: float) tuple
        """
        source_features = self.extract_speaker_embedding(source_audio, sr)
        output_features = self.extract_speaker_embedding(output_audio, sr)

        dist = float(np.linalg.norm(source_features - output_features))
        norm = float(np.linalg.norm(source_features) + np.linalg.norm(output_features))
        similarity = 1.0 - dist / (norm + 1e-8)

        leaked = similarity > threshold
        return leaked, similarity

    def _compute_spectral_fingerprint(
        self, audio: np.ndarray, sr: int, n_mels: int = 40
    ) -> np.ndarray:
        """Compute a spectral fingerprint for speaker identification."""
        if not LIBROSA_AVAILABLE:
            # Fallback: basic time-domain features
            return np.array(
                [
                    float(np.mean(audio)),
                    float(np.std(audio)),
                    float(np.max(np.abs(audio))),
                    float(np.sqrt(np.mean(audio**2))),
                ],
                dtype=np.float32,
            )

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=n_mels,
            n_fft=2048,
            hop_length=512,
        )
        fingerprint = np.mean(mel_spec, axis=1)
        centroid = float(
            np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        )
        rolloff = float(
            np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        )
        return np.concatenate([fingerprint, np.array([centroid, rolloff])])

    def _match_length(self, audio: np.ndarray, target_length: int) -> np.ndarray:
        """Ensure audio matches target length."""
        if len(audio) > target_length:
            return audio[:target_length]
        elif len(audio) < target_length:
            return np.pad(audio, (0, target_length - len(audio)))
        return audio


class SimpleVoiceConversionTransfer(IdentityTransfer):
    """Voice conversion using spectral envelope matching.

    Warps the target audio's spectral envelope toward the source
    speaker's characteristics using mel-frequency analysis. This is
    the default identity transfer method when no neural VC model is
    available.

    Accent leakage is detected by comparing source and output spectral
    fingerprints after transfer.
    """

    def __init__(
        self, n_mels: int = 40, frame_length: int = 2048, hop_length: int = 512
    ):
        self.n_mels = n_mels
        self.frame_length = frame_length
        self.hop_length = hop_length

    def transfer(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        target_audio: np.ndarray,
        target_sr: int,
        strength: float = 1.0,
    ) -> TransferResult:
        """Transfer source spectral envelope to target audio."""
        # Ensure same sample rate
        if source_sr != target_sr and LIBROSA_AVAILABLE:
            target_audio = librosa.resample(
                target_audio, orig_sr=target_sr, target_sr=source_sr
            )
            target_sr = source_sr

        target_len = len(target_audio)
        source_audio = self._match_length(source_audio, target_len)

        if strength <= 0:
            return TransferResult(
                waveform=target_audio.copy(),
                sample_rate=target_sr,
                modifications=[],
                source_features=self._compute_spectral_fingerprint(
                    source_audio, target_sr
                ),
                target_features=self._compute_spectral_fingerprint(
                    target_audio, target_sr
                ),
            )
            # Fallback: blend toward the US-realized target
            # (target already has US spectral characteristics, source has source identity)
            blended = (1 - strength) * target_audio + strength * source_audio
            return TransferResult(
                waveform=blended,
                sample_rate=target_sr,
                modifications=["time_domain_blend_fallback"],
                source_features=self._compute_spectral_fingerprint(
                    source_audio, target_sr
                ),
                target_features=self._compute_spectral_fingerprint(
                    target_audio, target_sr
                ),
            )

        if not LIBROSA_AVAILABLE:
            # Fallback: apply source's spectral shape to target via
            # magnitude-spectrum matching at frame level.
            # strength controls how aggressively source envelope replaces target envelope.
            from scipy.signal import stft as scipy_stft, istft as scipy_istft

            nperseg = 256
            f, t, D_src = scipy_stft(source_audio, nperseg=nperseg)
            _, _, D_tgt = scipy_stft(target_audio, nperseg=nperseg)

            min_frames = min(D_src.shape[1], D_tgt.shape[1])
            mag_src = np.abs(D_src[:, :min_frames])
            mag_tgt = np.abs(D_tgt[:, :min_frames])
            # Blend magnitude envelopes: source influence scales with strength
            matched_mag = strength * mag_src + (1 - strength) * mag_tgt
            # Use target phase (target has the correct pronunciation)
            D_out = matched_mag * np.exp(1j * np.angle(D_tgt[:, :min_frames]))
            _, output = scipy_istft(D_out, nperseg=nperseg)
            output = output.astype(np.float32)
            output = output[:len(target_audio)]
            if len(output) < len(target_audio):
                output = np.pad(output, (0, len(target_audio) - len(output)))
            return TransferResult(
                waveform=output,
                sample_rate=target_sr,
                modifications=["scipy_stft_magnitude_transfer"],
                source_features=self._compute_spectral_fingerprint(
                    source_audio, target_sr
                ),
                target_features=self._compute_spectral_fingerprint(
                    target_audio, target_sr
                ),
            )

        # Extract spectral envelopes
        source_mel = librosa.feature.melspectrogram(
            y=source_audio,
            sr=target_sr,
            n_mels=self.n_mels,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        source_env = np.mean(source_mel, axis=1)

        target_mel = librosa.feature.melspectrogram(
            y=target_audio,
            sr=target_sr,
            n_mels=self.n_mels,
            n_fft=self.frame_length,
            hop_length=self.hop_length,
        )
        target_env = np.mean(target_mel, axis=1)

        # Compute envelope ratio (source / target)
        ratio = source_env / (target_env + 1e-8)
        ratio = np.clip(ratio, 0.3, 3.0)

        # Map mel-domain ratio to full FFT spectrum
        mel_freqs = librosa.mel_frequencies(n_mels=self.n_mels)
        fft_freqs = librosa.fft_frequencies(sr=target_sr, n_fft=self.frame_length)

        from scipy.interpolate import interp1d

        interp = interp1d(
            mel_freqs, ratio, bounds_error=False, fill_value=1.0
        )
        full_ratio = np.clip(interp(fft_freqs), 0.3, 3.0)

        # Blend ratio based on strength
        blended_ratio = (1 - strength) * np.ones_like(full_ratio) + strength * full_ratio

        # Apply to target STFT
        D = librosa.stft(
            target_audio, n_fft=self.frame_length, hop_length=self.hop_length
        )
        mag = np.abs(D)
        phase = np.angle(D)

        mag_modified = mag * blended_ratio[:, np.newaxis]
        D_modified = mag_modified * np.exp(1j * phase)
        output = librosa.istft(
            D_modified, hop_length=self.hop_length, length=target_len
        )

        # Detect accent leakage
        leaked, confidence = self.detect_accent_leak(
            source_audio, output, target_sr
        )

        return TransferResult(
            waveform=output,
            sample_rate=target_sr,
            accent_leaked=leaked,
            leak_confidence=confidence,
            modifications=["spectral_envelope_transfer"],
            source_features=self._compute_spectral_fingerprint(
                source_audio, target_sr
            ),
            target_features=self._compute_spectral_fingerprint(
                target_audio, target_sr
            ),
        )

    def extract_speaker_embedding(
        self, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        """Extract spectral fingerprint as speaker embedding."""
        return self._compute_spectral_fingerprint(audio, sr)


class SpeakerEmbeddingTransfer(IdentityTransfer):
    """Identity transfer using speaker embeddings for conditioning.

    Prefers a neural VC backend (Seed-VC) when available. Falls back
    to SimpleVoiceConversionTransfer spectral matching otherwise.

    Tracks accent leakage through embedding similarity analysis.
    """

    def __init__(self, embedding_dim: int = 256):
        self.embedding_dim = embedding_dim
        self._seed_vc_wrapper = None
        self._load_attempted = False

    def _load_seed_vc(self):
        """Attempt to load Seed-VC wrapper if available."""
        if self._load_attempted:
            return
        self._load_attempted = True
        try:
            from seed_vc.seed_vc_wrapper import SeedVCWrapper

            self._seed_vc_wrapper = SeedVCWrapper()
            logger.info("Seed-VC wrapper loaded successfully")
        except Exception as e:
            logger.warning(
                "Seed-VC not available, using spectral fallback: %s", e
            )
            self._seed_vc_wrapper = None

    def transfer(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        target_audio: np.ndarray,
        target_sr: int,
        strength: float = 1.0,
    ) -> TransferResult:
        """Transfer identity using speaker embeddings."""
        self._load_seed_vc()

        if self._seed_vc_wrapper is not None and LIBROSA_AVAILABLE:
            try:
                return self._transfer_with_model(
                    source_audio, source_sr, target_audio, target_sr, strength
                )
            except Exception as e:
                logger.warning(
                    "Model-based transfer failed, using spectral fallback: %s", e
                )

        # Spectral fallback
        vc = SimpleVoiceConversionTransfer()
        return vc.transfer(source_audio, source_sr, target_audio, target_sr, strength)

    def _transfer_with_model(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        target_audio: np.ndarray,
        target_sr: int,
        strength: float,
    ) -> TransferResult:
        """Use Seed-VC for voice conversion."""
        import os
        import tempfile

        # Ensure target_sr matches source_sr for saving
        if source_sr != target_sr and LIBROSA_AVAILABLE:
            target_audio = librosa.resample(
                target_audio, orig_sr=target_sr, target_sr=source_sr
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            src_path = os.path.join(tmpdir, "source_vc.wav")
            tgt_path = os.path.join(tmpdir, "target_vc.wav")
            out_path = os.path.join(tmpdir, "output_vc.wav")

            sf.write(src_path, source_audio, source_sr)
            sf.write(tgt_path, target_audio, source_sr)

            self._seed_vc_wrapper.convert(src_path, tgt_path, out_path)

            if LIBROSA_AVAILABLE:
                output, _ = librosa.load(out_path, sr=source_sr)
            else:
                output, _ = sf.read(out_path, dtype=np.float32)
            output = output.astype(np.float32)

            # Blend based on strength
            min_len = min(len(output), len(target_audio))
            output = output[:min_len]
            tgt = target_audio[:min_len]
            output = (1 - strength) * tgt + strength * output

            leaked, confidence = self.detect_accent_leak(
                source_audio, output, source_sr
            )

            return TransferResult(
                waveform=output,
                sample_rate=source_sr,
                accent_leaked=leaked,
                leak_confidence=confidence,
                modifications=["seed_vc_voice_conversion"],
                source_features=self.extract_speaker_embedding(source_audio, source_sr),
                target_features=self.extract_speaker_embedding(target_audio, source_sr),
            )

    def extract_speaker_embedding(
        self, audio: np.ndarray, sr: int
    ) -> np.ndarray:
        """Extract speaker embedding using model or spectral fallback."""
        if self._seed_vc_wrapper is not None:
            try:
                return self._seed_vc_wrapper.extract_speaker_embedding(audio, sr)
            except Exception:
                pass

        return self._compute_spectral_fingerprint(audio, sr)

