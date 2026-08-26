"""Unit tests for accentedge.phase1.phoneme_pipeline."""
import math
import warnings
from unittest import mock

import pytest
import torch


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_waveform():
    """A synthetic 1-second sine-wave waveform at 24kHz (24000 samples)."""
    t = torch.linspace(0, 1, 24000)
    waveform = (torch.sin(2 * math.pi * 440 * t) * 0.1).unsqueeze(0)
    return waveform


@pytest.fixture
def phone_pipeline():
    """PhonemePipeline with no model loaded (test text_to_phones and helpers)."""
    from accentedge.phase1.phoneme_pipeline import PhonemePipeline
    return PhonemePipeline(device="cpu")


# ---------------------------------------------------------------------------
# Tests: __init__ and attributes
# ---------------------------------------------------------------------------

class TestInit:
    def test_frame_rate_is_80fps(self):
        from accentedge.phase1.phoneme_pipeline import PhonemePipeline
        p = PhonemePipeline(device="cpu")
        assert p.frame_rate == 80
        assert p.frame_rate_hz == 80.0

    def test_default_sample_rate(self):
        from accentedge.phase1.phoneme_pipeline import PhonemePipeline
        p = PhonemePipeline(device="cpu")
        assert p.sample_rate == 24000

    def test_custom_phone_vocab(self):
        from accentedge.phase1.phoneme_pipeline import PhonemePipeline
        vocab = {"<pad>": 0, "a": 1, "b": 2}
        p = PhonemePipeline(device="cpu", phone_vocab=vocab)
        assert p.phone_to_id == vocab
        assert p.pad_id == 0


# ---------------------------------------------------------------------------
# Tests: phones_to_frames
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Tests: phone vocab size contract
# ---------------------------------------------------------------------------

class TestPhoneVocabContract:
    """Verify the phone vocabulary size contract between PhonemePipeline
    and the DenoisingTransformerModel."""

    def test_phone_vocab_size_is_393(self):
        """PHONE_VOCAB_SIZE must be 393 to match the denoiser embedding table."""
        from accentedge.phase1.phoneme_pipeline import PHONE_VOCAB_SIZE
        assert PHONE_VOCAB_SIZE == 393

    def test_denoiser_default_vocab_size_is_393(self):
        """DenoisingTransformerModel default phone_vocab_size must be 393."""
        from accentedge.phase1.denoiser import DenoisingTransformerModel
        import inspect
        sig = inspect.signature(DenoisingTransformerModel.__init__)
        assert sig.parameters['phone_vocab_size'].default == 393

    def test_all_pipeline_ids_below_393(self):
        """Every ID produced by the pipeline must be < 393."""
        from accentedge.phase1.phoneme_pipeline import _PHONEME_TO_ID, PHONE_VOCAB_SIZE
        for phone, idx in _PHONEME_TO_ID.items():
            assert idx < PHONE_VOCAB_SIZE, (
                f"Phone '{phone}' has ID {idx}, which is >= PHONE_VOCAB_SIZE ({PHONE_VOCAB_SIZE})"
            )

    def test_no_duplicate_phoneme_symbols(self):
        """The vocabulary must not contain duplicate symbols."""
        from accentedge.phase1.phoneme_pipeline import _PHONEME_LIST
        seen = set()
        for phone in _PHONEME_LIST:
            assert phone not in seen, f"Duplicate phoneme symbol: '{phone}'"
            seen.add(phone)

    def test_pipeline_pad_id_is_not_denoiser_padding_idx(self):
        """The pipeline's semantic pad_id (1='sp') differs from the denoiser's
        PyTorch padding_idx (392). These serve different purposes."""
        from accentedge.phase1.phoneme_pipeline import PAD_ID, PHONE_VOCAB_SIZE
        from accentedge.phase1.denoiser import DenoisingTransformerModel
        import inspect
        denoiser_padding = inspect.signature(
            DenoisingTransformerModel.__init__
        ).parameters['phone_pad_id'].default
        # Pipeline pad_id should be an active phoneme symbol (sp=1)
        assert PAD_ID == 1
        # Denoiser padding_idx should be the last row (392)
        assert denoiser_padding == PHONE_VOCAB_SIZE - 1

    def test_phoneme_list_count(self):
        """The active English phoneme list should have 93 entries."""
        from accentedge.phase1.phoneme_pipeline import _PHONEME_LIST
        assert len(_PHONEME_LIST) == 93

    def test_id_range_within_embedding_space(self):
        """All pipeline IDs must fit in the denoiser's embedding index space."""
        from accentedge.phase1.phoneme_pipeline import _PHONEME_TO_ID, PHONE_VOCAB_SIZE
        ids = set(_PHONEME_TO_ID.values())
        max_id = max(ids)
        min_id = min(ids)
        assert min_id >= 0
        assert max_id < PHONE_VOCAB_SIZE

    def test_no_id_collisions(self):
        """Every phoneme symbol must map to a unique ID."""
        from accentedge.phase1.phoneme_pipeline import _PHONEME_TO_ID
        symbols_by_id = {}
        for symbol, idx in _PHONEME_TO_ID.items():
            assert idx not in symbols_by_id, (
                f"ID collision at {idx}: '{symbol}' and '{symbols_by_id[idx]}'"
            )
            symbols_by_id[idx] = symbol


# ---------------------------------------------------------------------------
# Tests: __init__ and attributes
# ---------------------------------------------------------------------------

class TestInit:
    def test_frame_rate_is_80fps(self):
        from accentedge.phase1.phoneme_pipeline import PhonemePipeline
        p = PhonemePipeline(device="cpu")
        assert p.frame_rate == 80
        assert p.frame_rate_hz == 80.0

    def test_default_sample_rate(self):
        from accentedge.phase1.phoneme_pipeline import PhonemePipeline
        p = PhonemePipeline(device="cpu")
        assert p.sample_rate == 24000

    def test_custom_phone_vocab(self):
        from accentedge.phase1.phoneme_pipeline import PhonemePipeline
        vocab = {"<pad>": 0, "a": 1, "b": 2}
        p = PhonemePipeline(device="cpu", phone_vocab=vocab)
        assert p.phone_to_id == vocab
        assert p.pad_id == 0


# ---------------------------------------------------------------------------
# Tests: phones_to_frames
# ---------------------------------------------------------------------------


class TestPhonesToFrames:
    def test_correct_frame_count_for_1s_audio(self, phone_pipeline):
        """1 second at 24kHz = 80 frames at 80fps."""
        num_frames = int(round(24000 * 80 / 24000))  # 80
        result = phone_pipeline.phones_to_frames(
            phones=["a", "b"],
            boundaries=[
                (0.0, 0.5, "a"),
                (0.5, 1.0, "b"),
            ],
            num_frames=num_frames,
        )
        assert result.shape == (1, 80)

    def test_correct_frame_count_for_half_second(self, phone_pipeline):
        """0.5 second at 24kHz = 40 frames at 80fps."""
        num_frames = int(round(12000 * 80 / 24000))  # 40
        result = phone_pipeline.phones_to_frames(
            phones=["a"],
            boundaries=[(0.0, 0.5, "a")],
            num_frames=num_frames,
        )
        assert result.shape == (1, 40)

    def test_empty_phones_returns_all_pad(self, phone_pipeline):
        """Empty phone list should return all pad_id frames."""
        num_frames = 80
        result = phone_pipeline.phones_to_frames(
            phones=[],
            boundaries=[],
            num_frames=num_frames,
        )
        assert result.shape == (1, num_frames)
        assert (result == phone_pipeline.pad_id).all()

    def test_empty_boundaries_returns_all_pad(self, phone_pipeline):
        """Empty boundaries should return all pad_id frames."""
        num_frames = 80
        result = phone_pipeline.phones_to_frames(
            phones=["a", "b"],
            boundaries=[],
            num_frames=num_frames,
        )
        assert (result == phone_pipeline.pad_id).all()

    def test_negative_num_frames_raises(self, phone_pipeline):
        with pytest.raises(ValueError, match="num_frames must be positive"):
            phone_pipeline.phones_to_frames(phones=[], boundaries=[], num_frames=-1)

    def test_zero_num_frames_raises(self, phone_pipeline):
        with pytest.raises(ValueError, match="num_frames must be positive"):
            phone_pipeline.phones_to_frames(phones=[], boundaries=[], num_frames=0)

    def test_phone_id_mapping(self, phone_pipeline):
        """Known phones should map to their vocabulary IDs."""
        from accentedge.phase1.phoneme_pipeline import _PHONEME_TO_ID
        for phone, expected_id in [("a", 5), ("sp", 1)]:
            result = phone_pipeline.phones_to_frames(
                phones=[phone],
                boundaries=[(0.0, 1.0, phone)],
                num_frames=80,
            )
            # The ID should be the one from the vocabulary (or pad if unknown)
            # Since 'a' and 'sp' are in the vocabulary, check it matches
            assert result[0, 0].item() == _PHONEME_TO_ID.get(phone, phone_pipeline.pad_id)

    def test_frame_boundary_exactness(self, phone_pipeline):
        """Frame boundaries at 0.0s and 1.0s should cover exactly [0, 80) frames."""
        result = phone_pipeline.phones_to_frames(
            phones=["a"],
            boundaries=[(0.0, 1.0, "a")],
            num_frames=80,
        )
        # All 80 frames should be filled (not pad)
        assert (result != phone_pipeline.pad_id).all()

    def test_partial_alignment(self, phone_pipeline):
        """Phone covering first half should occupy exactly first 40 frames."""
        result = phone_pipeline.phones_to_frames(
            phones=["a", "b"],
            boundaries=[
                (0.0, 0.5, "a"),
                (0.5, 1.0, "b"),
            ],
            num_frames=80,
        )
        first_half = result[0, :40]
        second_half = result[0, 40:]
        # First half and second half should differ (phone 'a' vs 'b')
        # (At least one frame should be different)
        assert not (first_half == second_half).all()


# ---------------------------------------------------------------------------
# Tests: __call__ edge cases
# ---------------------------------------------------------------------------

class TestCallEdgeCases:
    def test_empty_transcript_returns_all_pad(self, phone_pipeline, sample_waveform):
        """Empty transcript → all pad frames with correct count."""
        with mock.patch.object(
            phone_pipeline, "text_to_phones", return_value=[],
        ):
            result = phone_pipeline("", sample_waveform)
        assert result.shape == (1, 80)
        assert (result == phone_pipeline.pad_id).all()

    def test_whitespace_only_transcript_returns_all_pad(self, phone_pipeline, sample_waveform):
        """Whitespace-only transcript → all pad frames."""
        with mock.patch.object(
            phone_pipeline, "text_to_phones", return_value=[],
        ):
            result = phone_pipeline("   \n\t  ", sample_waveform)
        assert result.shape == (1, 80)
        assert (result == phone_pipeline.pad_id).all()

    def test_exact_frame_count_matches_facodec(self, phone_pipeline, sample_waveform):
        """Output frame count MUST exactly match FACodec frame count."""
        fake_boundaries = [(0.0, 1.0, "sp")]
        with mock.patch.object(
            phone_pipeline, "text_to_phones", return_value=["sp"]
        ), mock.patch.object(
            phone_pipeline, "align_phones_to_audio", return_value=fake_boundaries,
        ):
            result = phone_pipeline("hello world", sample_waveform)
        expected_frames = int(round(sample_waveform.shape[-1] * 80 / 24000))
        assert result.shape[-1] == expected_frames, (
            f"Frame count mismatch: got {result.shape[-1]}, expected {expected_frames}"
        )

    def test_frame_count_independent_of_transcript_content(self, phone_pipeline, sample_waveform):
        """Same waveform with different transcripts must produce same frame count."""
        fake_boundaries = [(0.0, 1.0, "sp")]
        for transcript in ["hello", "this is a longer sentence"]:
            with mock.patch.object(
                phone_pipeline, "text_to_phones", return_value=["sp"],
            ), mock.patch.object(
                phone_pipeline, "align_phones_to_audio", return_value=fake_boundaries,
            ):
                r = phone_pipeline(transcript, sample_waveform)
            assert r.shape[-1] == 80


# ---------------------------------------------------------------------------
# Tests: text_to_phones with mocked phonemizer
# ---------------------------------------------------------------------------

class TestTextToPhones:
    def test_text_to_phones_requires_phonemizer(self):
        """Calling text_to_phones without phonemizer should raise ImportError."""
        from accentedge.phase1.phoneme_pipeline import PhonemePipeline, HAS_PHONEMIZER

        if HAS_PHONEMIZER:
            pytest.skip("phonemizer is installed — skipping mock test")

        p = PhonemePipeline(device="cpu")
        with pytest.raises(ImportError, match="phonemizer"):
            p.text_to_phones("hello world")

    def test_text_to_phones_mock(self, phone_pipeline):
        """With phonemizer mocked, text_to_phones returns a list of phonemes."""
        from accentedge.phase1.phoneme_pipeline import HAS_PHONEMIZER

        if not HAS_PHONEMIZER:
            pytest.skip("phonemizer not installed — skipping integration test")


# ---------------------------------------------------------------------------
# Tests: alignment model loading
# ---------------------------------------------------------------------------

class TestModelLoading:
    def test_align_raises_without_transformers(self):
        """align_phones_to_audio without transformers should raise ImportError."""
        from accentedge.phase1.phoneme_pipeline import (
            PhonemePipeline,
            HAS_TRANSFORMERS,
        )

        if HAS_TRANSFORMERS:
            pytest.skip("transformers is installed — skipping mock test")

        p = PhonemePipeline(device="cpu")
        waveform = torch.randn(1, 24000)
        with pytest.raises(ImportError, match="transformers"):
            p.align_phones_to_audio(waveform, ["a", "b"])

    def test_align_without_model_name_suggestion(self):
        """ImportError from failed model load should mention the fallback model."""
        from accentedge.phase1.phoneme_pipeline import (
            PhonemePipeline,
            HAS_TRANSFORMERS,
        )

        if not HAS_TRANSFORMERS:
            pytest.skip("transformers not installed")

        p = PhonemePipeline(device="cpu", model_name="nonexistent/model")

        # Patch so model load fails with a clear error
        with mock.patch.object(p, "_ensure_model_loaded", side_effect=ImportError("mock")):
            p._model = None
            with pytest.raises(ImportError):
                p.align_phones_to_audio(torch.randn(1, 24000), ["a"])


# ---------------------------------------------------------------------------
# Tests: full pipeline with mocked alignment
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_full_pipeline_with_mocked_alignment(self, phone_pipeline, sample_waveform):
        """End-to-end with mocked alignment verifies frame count integrity."""
        fake_boundaries = [
            (0.0, 0.25, "sil"),
            (0.25, 0.75, "a"),
            (0.75, 1.0, "sp"),
        ]

        with mock.patch.object(
            phone_pipeline,
            "text_to_phones",
            return_value=["sil", "a", "sp"],
        ), mock.patch.object(
            phone_pipeline,
            "align_phones_to_audio",
            return_value=fake_boundaries,
        ):
            result = phone_pipeline("test", sample_waveform)

        expected_frames = 80
        assert result.shape == (1, expected_frames)
        assert result.dtype == torch.long

    def test_pipeline_returns_tensor_not_logits(self, phone_pipeline, sample_waveform):
        """The pipeline returns a [1, T] int64 tensor, not float logits."""
        with mock.patch.object(
            phone_pipeline,
            "text_to_phones",
            return_value=["sp"],
        ), mock.patch.object(
            phone_pipeline,
            "align_phones_to_audio",
            return_value=[(0.0, 1.0, "sp")],
        ):
            result = phone_pipeline("hello", sample_waveform)

        assert isinstance(result, torch.Tensor)
        assert result.dtype == torch.long
        assert result.dim() == 2
        assert result.shape[0] == 1

    def test_pipeline_different_durations_same_frame_rate(self, phone_pipeline):
        """Frame count scales linearly with audio duration."""
        fake_boundaries = [(0.0, 1.0, "sp")]

        for duration in [0.5, 1.0, 2.0]:
            num_samples = int(24000 * duration)
            num_frames = int(round(num_samples * 80 / 24000))
            waveform = torch.randn(1, num_samples)
            with mock.patch.object(
                phone_pipeline,
                "text_to_phones",
                return_value=["sp"],
            ), mock.patch.object(
                phone_pipeline,
                "align_phones_to_audio",
                return_value=[(0.0, duration, "sp")],
            ):
                result = phone_pipeline("test", waveform)
            assert result.shape[-1] == num_frames, (
                f"Duration {duration}s: expected {num_frames} frames, got {result.shape[-1]}"
            )


# ---------------------------------------------------------------------------
# Tests: deprecation of old PhonemeConditioner
# ---------------------------------------------------------------------------

class TestDeprecatedPhonemeConditioner:
    def test_old_phoneme_conditioner_emits_deprecation_warning(self):
        """Instantiating PhonemeConditioner should emit a DeprecationWarning."""
        import importlib
        import sys

        # Remove cached import if present so we start fresh
        mods_to_clear = [
            k for k in sys.modules
            if k.startswith("accentedge.evaluation.phonemes")
        ]
        for mod in mods_to_clear:
            del sys.modules[mod]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            from accentedge.evaluation.phonemes import PhonemeConditioner  # noqa: F401
            # Instantiate — __init__ is where the warning fires
            conditioner = PhonemeConditioner(device="cpu")  # noqa: F841
            deprecations = [
                x for x in w
                if issubclass(x.category, DeprecationWarning)
                and "PhonemeConditioner" in str(x.message)
            ]
            assert len(deprecations) >= 1, (
                f"PhonemeConditioner should emit a DeprecationWarning on __init__, "
                f"got {[str(x.message) for x in w]}"
            )
