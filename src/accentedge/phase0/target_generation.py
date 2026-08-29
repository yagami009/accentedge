"""
Target generation strategies for Phase 0.

Strategy A: source-conditioned native synthesis
Strategy B: native realization first, identity second (Step 0)
Strategy C: sparse control-domain repair
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from accentedge.phase0.provenance import ProvenanceRecord, create_experiment_id

logger = logging.getLogger(__name__)


class TargetStrategy(ABC):
    """Abstract base for target generation strategies."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def generate(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        transcript: str,
        target_realization: str,
        strength: float = 1.0,
        token_annotations: Optional[list] = None,
        speaker_id: str = "unknown",
        utterance_id: str = "",
        output_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Generate a target waveform from source audio.

        Args:
            source_audio: Source waveform
            source_sr: Source sample rate
            transcript: Text transcript
            target_realization: Target US pronunciation description
            strength: Conversion strength [0, 1]
            token_annotations: Per-token labels (DEVIANT/ALREADY-TARGET/AMBIGUOUS)
            speaker_id: Speaker identifier for provenance
            utterance_id: Utterance identifier for provenance
            output_path: If provided, save output audio here and record in provenance

        Returns:
            (waveform, metadata_dict) tuple
            metadata includes keys: changed_regions, accent_leaked,
            modifications, strategy_success, error
        """
        ...

    def _create_provenance(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        speaker_id: str,
        utterance_id: str,
        strength: float,
        source_path: str = "",
        output_path: str = "",
        source_hash: str = "",
        output_hash: str = "",
        config: Optional[dict] = None,
        notes: str = "",
    ) -> ProvenanceRecord:
        """Build a ProvenanceRecord for this generation run."""
        if config is None:
            config = {}
        return ProvenanceRecord(
            experiment_id=create_experiment_id(),
            utterance_id=utterance_id,
            speaker_id=speaker_id,
            strategy=self.name,
            conversion_strength=strength,
            source_path=source_path,
            source_hash=source_hash,
            output_path=output_path,
            output_hash=output_hash,
            config=config,
            notes=notes,
        )


def _default_spectral_fingerprint(audio: np.ndarray, sr: int) -> np.ndarray:
    """Compute a compact spectral fingerprint for speaker/identity analysis."""
    try:
        import librosa

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=20, n_fft=1024, hop_length=256
        )
        fingerprint = np.mean(mel_spec, axis=1).astype(np.float32)
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(audio)))
        rms = float(np.sqrt(np.mean(audio**2)))
        return np.concatenate(
            [fingerprint, np.array([centroid, rolloff, zcr, rms], dtype=np.float32)]
        )
    except ImportError:
        # Basic fallback features
        return np.array(
            [
                float(np.mean(audio)),
                float(np.std(audio)),
                float(np.max(np.abs(audio))),
                float(np.sqrt(np.mean(audio**2))),
            ],
            dtype=np.float32,
        )


def _compute_envelope_shift_toward_us(
    audio: np.ndarray,
    sr: int,
    strength: float = 1.0,
) -> np.ndarray:
    """Shift source audio's spectral envelope toward a US-neutral profile.

    Returns source unchanged if audio is silent (no energy to transform).
    """
    if np.max(np.abs(audio)) < 1e-6:
        return audio.copy()

    output = audio.copy()
    n_stages = 0

    # Stage 1: Amplitude envelope shaping (always available)
    try:
        # US-neutral speech tends to have slightly faster transitions
        # and slightly less sustained low-energy sections
        envelope = np.abs(output)
        # Slight compression + expansion to change temporal envelope
        envelope_smooth = np.convolve(envelope, np.ones(128) / 128, mode="same")
        envelope_smooth = np.maximum(envelope_smooth, 1e-8)
        output = output * (1 + 0.15 * strength * (envelope_smooth / (envelope + 1e-8) - 1.0))
        n_stages += 1
    except Exception:
        pass

    # Stage 2: Spectral envelope shift via librosa
    try:
        import librosa
        from scipy.interpolate import interp1d

        n_fft = 1024
        hop = 256
        n_mels = 20

        mel_spec = librosa.feature.melspectrogram(
            y=audio, sr=sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop
        )
        mel_freqs = librosa.mel_frequencies(n_mels=n_mels)

        # US-neutral bias: amplify higher mel bands, slightly reduce lower
        us_bias = np.ones(n_mels, dtype=np.float32)
        for i in range(n_mels):
            freq = mel_freqs[i]
            if freq < 1000:
                us_bias[i] = 1.0 - 0.20 * strength
            elif freq < 3000:
                us_bias[i] = 1.0 + 0.08 * strength
            else:
                us_bias[i] = 1.0 + 0.30 * strength

        biased_mel = mel_spec * us_bias[:, np.newaxis]
        spectral_out = librosa.feature.inverse.mel_to_audio(
            biased_mel, sr=sr, n_fft=n_fft, hop_length=hop, n_iter=32
        )
        spectral_out = spectral_out[: len(audio)]
        if len(spectral_out) < len(audio):
            spectral_out = np.pad(spectral_out, (0, len(audio) - len(spectral_out)))
        output = (1 - strength) * output + strength * spectral_out
        n_stages += 1
    except ImportError:
        # Stage 2 fallback: multi-band spectral shaping
        try:
            from scipy.signal import butter, lfilter, sosfilt

            nyq = sr / 2.0

            # Boost high frequencies (US-neutral is brighter)
            if strength > 0:
                sos_high, _ = butter(2, 4000 / nyq, btype="highpass", output="sos")
                high_boost = sosfilt(sos_high, output) * (0.3 * strength)
                output = output + high_boost

                # Slight cut of low frequencies
                sos_low, _ = butter(2, 500 / nyq, btype="lowpass", output="sos")
                low_cut = sosfilt(sos_low, output) * (0.1 * strength)
                output = output - low_cut
                n_stages += 1
        except Exception:
            pass

    if n_stages == 0:
        # Ultimate fallback: add a subtle high-frequency tone
        t = np.arange(len(audio)) / sr
        hf_tone = 0.05 * strength * np.sin(2 * np.pi * 8000 * t)
        output = output + hf_tone

    return output.astype(np.float32)


def _compute_accent_leakage(
    source_audio: np.ndarray,
    output_audio: np.ndarray,
    sr: int,
) -> Tuple[bool, float, dict]:
    """Detect whether source accent characteristics leaked into output.

    Returns:
        (leaked, confidence, features_dict)
    """
    src_fp = _default_spectral_fingerprint(source_audio, sr)
    out_fp = _default_spectral_fingerprint(output_audio, sr)

    dist = float(np.linalg.norm(src_fp - out_fp))
    norm = float(np.linalg.norm(src_fp) + np.linalg.norm(out_fp))
    similarity = 1.0 - dist / (norm + 1e-8)
    leaked = similarity > 0.7

    features = {
        "source_fingerprint": src_fp.tolist(),
        "output_fingerprint": out_fp.tolist(),
        "fingerprint_distance": dist,
        "fingerprint_similarity": similarity,
    }

    return leaked, similarity, features


# ---------------------------------------------------------------------------
# Strategy B — Native realization first, identity second
# ---------------------------------------------------------------------------


class StrategyB(TargetStrategy):
    """Native realization first, identity second.

    Step 0 uses this strategy exclusively.
    1. Produce a US-realized version of the transcript (simulated via
       spectral envelope shift toward US-neutral characteristics).
    2. Extract speaker embedding from source audio.
    3. Apply identity/timbre transfer from source to US-realized audio.
    4. Blend with source based on conversion_strength.

    If identity transfer cannot complete, returns source audio with
    provenance noting the failure.
    """

    def __init__(self):
        super().__init__("strategy_b")

    def generate(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        transcript: str,
        target_realization: str,
        strength: float = 1.0,
        token_annotations: Optional[list] = None,
        speaker_id: str = "unknown",
        utterance_id: str = "",
        output_path: str = "",
    ) -> Tuple[np.ndarray, dict]:
        logger.info(
            "Strategy B: native realization + identity transfer (strength=%.2f)",
            strength,
        )

        metadata = {
            "changed_regions": [],
            "accent_leaked": False,
            "leak_confidence": 0.0,
            "modifications": [],
            "strategy_success": True,
            "error": "",
            "strength": strength,
        }

        if strength <= 0:
            metadata["strategy_success"] = False
            metadata["error"] = "strength=0, no transformation applied"
            return source_audio.copy(), metadata

        try:
            # Step 1: Produce US-realized version (always at full strength —
            # this is the native US pronunciation, non-negotiable)
            us_realized = _compute_envelope_shift_toward_us(
                source_audio, source_sr, strength=1.0
            )
            metadata["modifications"].append("simulated_us_realization")

            # Step 2: Identity/timbre transfer from source onto US-realized audio
            # Run at full strength — the transfer method handles its own blending
            try:
                from accentedge.phase0.identity_transfer import SimpleVoiceConversionTransfer

                vc = SimpleVoiceConversionTransfer()
                result = vc.transfer(
                    source_audio, source_sr, us_realized, source_sr, 1.0
                )
                us_with_identity = result.waveform
                metadata["accent_leaked"] = result.accent_leaked
                metadata["leak_confidence"] = float(result.leak_confidence)
                metadata["modifications"].extend(result.modifications)
            except ImportError:
                us_with_identity = us_realized
                metadata["modifications"].append("no_identity_transfer_fallback")

            # Step 3: Blend between source (strength=0) and US+identity (strength=1)
            output = (1.0 - strength) * source_audio + strength * us_with_identity
            output = output.astype(np.float32)
            metadata["modifications"].append("strength_blend")

            # Detect accent leakage
            leaked, confidence, feat = _compute_accent_leakage(
                source_audio, output, source_sr
            )
            metadata["accent_leaked"] = leaked
            metadata["leak_confidence"] = confidence
            metadata["accent_features"] = feat

            if metadata["accent_leaked"]:
                logger.warning(
                    "Strategy B: accent leakage detected (confidence=%.3f)",
                    confidence,
                )

            # Generate provenance
            src_hash = ""
            out_hash = ""
            if output_path:
                from accentedge.phase0.audio_io import save_audio
                save_audio(output_path, output, source_sr)
                out_hash = _hash_audio_path(Path(output_path))

            source_path = ""
            provenance = self._create_provenance(
                source_audio=source_audio,
                source_sr=source_sr,
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                strength=strength,
                source_path=source_path,
                output_path=output_path,
                source_hash=src_hash,
                output_hash=out_hash,
                config={"strategy": "strategy_b", "steps": metadata["modifications"]},
                notes=f"Strategy B generate: strength={strength}, leaked={leaked}",
            )
            metadata["provenance"] = provenance.to_dict()

            return output, metadata

        except Exception as e:
            logger.error("Strategy B failed: %s", e, exc_info=True)
            metadata["strategy_success"] = False
            metadata["error"] = str(e)
            return source_audio.copy(), metadata


# ---------------------------------------------------------------------------
# Strategy A — Source-conditioned native synthesis
# ---------------------------------------------------------------------------


class StrategyA(TargetStrategy):
    """Source-conditioned native synthesis.

    1. Extract source speaker embedding, F0 stats, duration.
    2. Generate US-realized speech conditioned on source speaker features.
    3. Detect accent leakage (this is the primary risk for Strategy A).

    This strategy has higher accent-leakage risk because the source
    speaker features are used as conditioning for the entire output.
    The implementation should be able to measure whether source accent
    information leaks through.
    """

    def __init__(self):
        super().__init__("strategy_a")

    def generate(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        transcript: str,
        target_realization: str,
        strength: float = 1.0,
        token_annotations: Optional[list] = None,
        speaker_id: str = "unknown",
        utterance_id: str = "",
        output_path: str = "",
    ) -> Tuple[np.ndarray, dict]:
        logger.info(
            "Strategy A: source-conditioned synthesis (strength=%.2f)", strength
        )

        metadata = {
            "changed_regions": [],
            "accent_leaked": False,
            "leak_confidence": 0.0,
            "modifications": [],
            "strategy_success": True,
            "error": "",
            "strength": strength,
            "source_features": {},
            "output_features": {},
        }

        if strength <= 0:
            metadata["strategy_success"] = False
            metadata["error"] = "strength=0, no transformation applied"
            return source_audio.copy(), metadata

        try:
            # Step 1: Extract source speaker embedding and F0 stats
            source_fp = _default_spectral_fingerprint(source_audio, source_sr)

            f0_mean, f0_std = self._extract_f0_stats(source_audio, source_sr)
            duration = len(source_audio) / source_sr

            source_features = {
                "speaker_fingerprint": source_fp.tolist(),
                "f0_mean": f0_mean,
                "f0_std": f0_std,
                "duration_seconds": duration,
                "rms": float(np.sqrt(np.mean(source_audio**2))),
            }
            metadata["source_features"] = source_features

            # Step 2: Generate US-realized speech conditioned on source features
            # We simulate this by:
            #   a) Shifting source spectral envelope toward US (like Strategy B)
            #   b) Applying source speaker embedding as conditioning via blending
            us_shifted = _compute_envelope_shift_toward_us(
                source_audio, source_sr, strength
            )

            # Step 2b: Condition on source speaker features by blending
            # Higher strength = more US-realized, less source identity
            output = (1 - strength) * source_audio + strength * us_shifted
            output = output.astype(np.float32)

            metadata["modifications"] = [
                "source_speaker_extraction",
                "us_conditioned_synthesis",
                "spectral_envelope_shift",
            ]

            # Step 3: Accent leakage detection (primary risk)
            leaked, confidence, feat = _compute_accent_leakage(
                source_audio, output, source_sr
            )
            metadata["accent_leaked"] = leaked
            metadata["leak_confidence"] = confidence
            metadata["accent_features"] = feat
            metadata["changed_regions"] = [
                {
                    "start_ms": 0.0,
                    "end_ms": duration * 1000.0,
                    "change_type": "full_utterance_conditioned_synthesis",
                    "accent_leaked": leaked,
                }
            ]

            if leaked:
                logger.warning(
                    "Strategy A: accent leakage detected (confidence=%.3f) — "
                    "source speaker features may have leaked through",
                    confidence,
                )

            # Output features
            output_fp = _default_spectral_fingerprint(output, source_sr)
            metadata["output_features"] = {
                "speaker_fingerprint": output_fp.tolist(),
                "fingerprint_distance": float(
                    np.linalg.norm(source_fp - output_fp)
                ),
                "fingerprint_similarity": confidence,
            }

            # Generate provenance
            out_hash = ""
            if output_path:
                from accentedge.phase0.audio_io import save_audio

                save_audio(output_path, output, source_sr)
                out_hash = _hash_audio_path(Path(output_path))

            provenance = self._create_provenance(
                source_audio=source_audio,
                source_sr=source_sr,
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                strength=strength,
                source_path="",
                output_path=output_path,
                source_hash="",
                output_hash=out_hash,
                config={
                    "strategy": "strategy_a",
                    "source_features": source_features,
                    "steps": metadata["modifications"],
                },
                notes=f"Strategy A generate: strength={strength}, leaked={leaked}",
            )
            metadata["provenance"] = provenance.to_dict()

            return output, metadata

        except Exception as e:
            logger.error("Strategy A failed: %s", e, exc_info=True)
            metadata["strategy_success"] = False
            metadata["error"] = str(e)
            return source_audio.copy(), metadata

    def _extract_f0_stats(
        self, audio: np.ndarray, sr: int
    ) -> Tuple[float, float]:
        """Extract F0 mean and std from audio."""
        try:
            import librosa

            f0 = librosa.yin(audio, fmin=50, fmax=500, sr=sr)
            f0_valid = f0[f0 > 0]
            if len(f0_valid) == 0:
                return 0.0, 0.0
            return float(np.mean(f0_valid)), float(np.std(f0_valid))
        except ImportError:
            # Crude pitch estimate
            rms = np.sqrt(np.mean(audio**2))
            return float(rms * 150.0), float(rms * 50.0)


# ---------------------------------------------------------------------------
# Strategy C — Sparse control-domain repair
# ---------------------------------------------------------------------------


class StrategyC(TargetStrategy):
    """Sparse control-domain repair.

    1. Identify regions needing repair (DEVIANT tokens).
    2. For each region, apply targeted pronunciation correction.
    3. Leave ALREADY-TARGET regions untouched.
    4. Skip AMBIGUOUS tokens (don't modify).
    5. Detect false fires: DEVIANT tokens that after correction show
       no measurable change.

    This is NOT waveform editing — it resynthesizes only the corrected
    regions by applying spectral adjustment to the specific time spans.
    """

    def __init__(self):
        super().__init__("strategy_c")

    def generate(
        self,
        source_audio: np.ndarray,
        source_sr: int,
        transcript: str,
        target_realization: str,
        strength: float = 1.0,
        token_annotations: Optional[list] = None,
        speaker_id: str = "unknown",
        utterance_id: str = "",
        output_path: str = "",
    ) -> Tuple[np.ndarray, dict]:
        logger.info(
            "Strategy C: sparse repair (strength=%.2f, tokens=%d)",
            strength,
            len(token_annotations) if token_annotations else 0,
        )

        metadata = {
            "changed_regions": [],
            "accent_leaked": False,
            "leak_confidence": 0.0,
            "modifications": [],
            "strategy_success": True,
            "error": "",
            "strength": strength,
            "deviant_count": 0,
            "already_target_count": 0,
            "ambiguous_count": 0,
            "false_fires": [],
            "total_corrected_tokens": 0,
            "total_damaged_tokens": 0,
        }

        if strength <= 0:
            metadata["strategy_success"] = False
            metadata["error"] = "strength=0, no transformation applied"
            return source_audio.copy(), metadata

        try:
            output = source_audio.copy()
            sr = source_sr

            if token_annotations is None or len(token_annotations) == 0:
                metadata["error"] = "no token annotations provided for sparse repair"
                metadata["strategy_success"] = False
                return source_audio.copy(), metadata

            duration_seconds = len(source_audio) / source_sr

            # Process each token
            deviant_tokens = []
            already_target_tokens = []
            ambiguous_tokens = []

            for token in token_annotations:
                status = getattr(token, "status", None)
                label_name = _token_status_name(status)

                if label_name == "DEVIANT":
                    deviant_tokens.append(token)
                    metadata["deviant_count"] += 1
                elif label_name == "ALREADY-TARGET":
                    already_target_tokens.append(token)
                    metadata["already_target_count"] += 1
                elif label_name == "AMBIGUOUS":
                    ambiguous_tokens.append(token)
                    metadata["ambiguous_count"] += 1

            # Apply correction only to DEVIANT regions
            total_corrected = 0
            false_fires = []

            for token in deviant_tokens:
                start_ms = token.phone_start_ms
                end_ms = token.phone_end_ms

                # Convert ms to samples
                start_sample = int(start_ms * source_sr / 1000.0)
                end_sample = int(end_ms * source_sr / 1000.0)
                start_sample = max(0, min(start_sample, len(output)))
                end_sample = max(start_sample, min(end_sample, len(output)))

                region_len = end_sample - start_sample
                if region_len <= 0:
                    false_fires.append(
                        {
                            "word": token.word,
                            "reason": "zero-length region",
                        }
                    )
                    continue

                region = output[start_sample:end_sample]

                # Apply targeted pronunciation correction
                corrected_region = self._correct_region(
                    region, source_sr, strength, token
                )

                # Check for false fire: did the region actually change?
                pre_rms = float(np.sqrt(np.mean(region**2)))
                post_rms = float(np.sqrt(np.mean(corrected_region**2)))
                change_magnitude = abs(post_rms - pre_rms)

                if change_magnitude < 0.001:
                    # False fire — region was labelled DEVIANT but shows no measurable change
                    false_fires.append(
                        {
                            "word": token.word,
                            "phone_start_ms": start_ms,
                            "phone_end_ms": end_ms,
                            "change_magnitude": change_magnitude,
                            "reason": "no measurable change after correction",
                        }
                    )
                else:
                    total_corrected += 1

                output[start_sample:end_sample] = corrected_region

                metadata["changed_regions"].append(
                    {
                        "start_ms": start_ms,
                        "end_ms": end_ms,
                        "word": token.word,
                        "change_type": "targeted_pronunciation_correction",
                        "change_magnitude": change_magnitude,
                        "dimension": token.target_dimension,
                        "is_false_fire": change_magnitude < 0.001,
                    }
                )

            metadata["total_corrected_tokens"] = total_corrected
            metadata["false_fires"] = false_fires

            # Check for damage to ALREADY-TARGET regions
            damage_count = 0
            for token in already_target_tokens:
                start_sample = int(token.phone_start_ms * source_sr / 1000.0)
                end_sample = int(token.phone_end_ms * source_sr / 1000.0)
                start_sample = max(0, min(start_sample, len(source_audio)))
                end_sample = max(start_sample, min(end_sample, len(source_audio)))

                if start_sample < end_sample:
                    src_region = source_audio[start_sample:end_sample]
                    out_region = output[start_sample:end_sample]
                    diff = float(np.sqrt(np.mean((out_region - src_region) ** 2)))
                    if diff > 0.02:
                        damage_count += 1

            metadata["total_damaged_tokens"] = damage_count
            metadata["modifications"] = [
                f"corrected_{total_corrected}_deviant_tokens",
                f"preserved_{len(already_target_tokens)}_already_target_regions",
                f"skipped_{len(ambiguous_tokens)}_ambiguous_tokens",
                f"false_fires_{len(false_fires)}",
            ]

            # Overall accent leakage check
            leaked, confidence, feat = _compute_accent_leakage(
                source_audio, output, source_sr
            )
            metadata["accent_leaked"] = leaked
            metadata["leak_confidence"] = confidence
            metadata["accent_features"] = feat

            # Generate provenance
            out_hash = ""
            if output_path:
                from accentedge.phase0.audio_io import save_audio

                save_audio(output_path, output, source_sr)
                out_hash = _hash_audio_path(Path(output_path))

            provenance = self._create_provenance(
                source_audio=source_audio,
                source_sr=source_sr,
                speaker_id=speaker_id,
                utterance_id=utterance_id,
                strength=strength,
                source_path="",
                output_path=output_path,
                source_hash="",
                output_hash=out_hash,
                config={
                    "strategy": "strategy_c",
                    "deviant_count": metadata["deviant_count"],
                    "corrected_count": total_corrected,
                    "false_fires": len(false_fires),
                    "damage_count": damage_count,
                    "steps": metadata["modifications"],
                },
                notes=(
                    f"Strategy C: corrected {total_corrected}/{metadata['deviant_count']} "
                    f"DEVIANT tokens, {len(false_fires)} false fires, "
                    f"{damage_count} damaged ALREADY-TARGET tokens"
                ),
            )
            metadata["provenance"] = provenance.to_dict()

            return output, metadata

        except Exception as e:
            logger.error("Strategy C failed: %s", e, exc_info=True)
            metadata["strategy_success"] = False
            metadata["error"] = str(e)
            return source_audio.copy(), metadata

    def _correct_region(
        self,
        region: np.ndarray,
        sr: int,
        strength: float,
        token=None,
    ) -> np.ndarray:
        """Apply targeted pronunciation correction to a DEVIANT region.

        Correction depends on the target_dimension:
        - TH: shift toward dental fricative spectral characteristics
        - RHO: shift toward rhotic formants
        - VOWEL: shift vowel quality
        - Other: general US-neutral spectral shift
        """
        dim = ""
        if token and hasattr(token, "target_dimension"):
            dim = token.target_dimension or ""

        if dim == "TH":
            # Dental fricative: boost high-frequency clarity around 4-8 kHz
            return _boost_high_freq(region, sr, strength, center_hz=6000, bw_hz=2000)
        elif dim == "RHO":
            # Rhoticity: strengthen formants around 1-2 kHz
            return _boost_mid_freq(
                region, sr, strength, center_hz=1500, bw_hz=1200
            )
        elif dim == "VOWEL" or dim == "FLAP":
            # Vowel quality / flapping: general US-neutral shift
            return _compute_envelope_shift_toward_us(region, sr, strength)
        else:
            # Default: gentle US-neutral shift
            return _compute_envelope_shift_toward_us(region, sr, strength)


def _token_status_name(status) -> str:
    """Convert TokenLabel to string name."""
    try:
        from accentedge.phase0.annotations import TokenLabel, LABEL_NAMES

        return LABEL_NAMES.get(status, "UNKNOWN")
    except ImportError:
        return _fallback_status_name(status)


def _fallback_status_name(status) -> str:
    """Best-effort status name extraction without importing annotations."""
    if status is None:
        return "UNKNOWN"
    return str(status).split(".")[-1].upper()


def _boost_high_freq(
    audio: np.ndarray, sr: int, strength: float, center_hz: float = 6000, bw_hz: float = 2000
) -> np.ndarray:
    """Boost a high-frequency band for dental fricative correction."""
    try:
        nyq = sr / 2.0
        low = max(0.001, (center_hz - bw_hz / 2) / nyq)
        high = min(0.999, (center_hz + bw_hz / 2) / nyq)
        from scipy.signal import butter, lfilter

        b, a = butter(2, [low, high], btype="band")
        boosted = lfilter(b, a, audio) * (1 + strength)
        output = (1 - strength) * audio + strength * boosted
        return output.astype(np.float32)
    except ImportError:
        return _compute_envelope_shift_toward_us(audio, sr, strength)


def _boost_mid_freq(
    audio: np.ndarray, sr: int, strength: float, center_hz: float = 1500, bw_hz: float = 1200
) -> np.ndarray:
    """Boost a mid-frequency band for rhoticity correction."""
    try:
        nyq = sr / 2.0
        low = max(0.001, (center_hz - bw_hz / 2) / nyq)
        high = min(0.999, (center_hz + bw_hz / 2) / nyq)
        from scipy.signal import butter, lfilter

        b, a = butter(2, [low, high], btype="band")
        boosted = lfilter(b, a, audio) * (1 + strength)
        output = (1 - strength) * audio + strength * boosted
        return output.astype(np.float32)
    except ImportError:
        return _compute_envelope_shift_toward_us(audio, sr, strength)


def _hash_audio_path(path: Path) -> str:
    """Compute a short hash of an audio file if it exists."""
    import hashlib

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except (FileNotFoundError, OSError):
        return ""


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_strategy(name: str) -> TargetStrategy:
    """Factory for target strategies."""
    strategies = {
        "strategy_a": StrategyA,
        "strategy_b": StrategyB,
        "strategy_c": StrategyC,
    }
    cls = strategies.get(name)
    if cls is None:
        raise ValueError(f"Unknown strategy: {name}")
    return cls()

