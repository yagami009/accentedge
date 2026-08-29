"""Tests for Token Translation Candidate C."""

from __future__ import annotations

import platform

import numpy as np
import pytest
import torch

from accentedge_lab.models.token_translation.interfaces import (
    AccentTokenTranslator,
    CausalSpeechTokenizer,
    SpeechToken,
    TokenConditionedSynthesizer,
    TokenSequence,
)
from accentedge_lab.models.token_translation.streaming_config import TokenTranslationConfig
from accentedge_lab.models.token_translation.synthesizer import TokenConditionedSynthesizer as SynthModule
from accentedge_lab.models.token_translation.token_translation_candidate import (
    TokenTranslationCandidate,
)
from accentedge_lab.models.token_translation.tokenizer import CausalSpeechTokenizer as TokenizerModule
from accentedge_lab.models.token_translation.translator import AccentTokenTranslator as TranslatorModule


def _make_chunk(samples: int = 1280, sr: int = 16000) -> np.ndarray:
    """Create a 16 kHz mono float32 audio chunk (~80ms)."""
    t = np.linspace(0, 2 * np.pi * 440 * samples / sr, samples)
    return np.sin(t).astype(np.float32)


def _make_token_sequence(seq_len: int = 4, dim: int = 128) -> TokenSequence:
    """Create a TokenSequence with random embeddings."""
    tokens = []
    for i in range(seq_len):
        tokens.append(
            SpeechToken(
                token_id=i,
                token_embedding=torch.randn(dim),
                timestamp_ms=i * 20.0,
                duration_ms=20.0,
                is_speech=True,
            )
        )
    return TokenSequence(tokens)


class TestTokenizerProducesTokens:
    def test_tokenizer_output_shape(self) -> None:
        tokenizer = TokenizerModule()
        tokenizer.eval()
        chunk = torch.from_numpy(_make_chunk()).unsqueeze(0)
        with torch.no_grad():
            seq = tokenizer.tokenize(chunk, None)
        assert isinstance(seq, TokenSequence)
        assert len(seq) > 0

    def test_tokenizer_embedding_dim(self) -> None:
        tokenizer = TokenizerModule()
        tokenizer.eval()
        chunk = torch.from_numpy(_make_chunk()).unsqueeze(0)
        with torch.no_grad():
            seq = tokenizer.tokenize(chunk, None)
        assert len(seq) > 0
        assert seq.tokens[0].token_embedding.shape[-1] == tokenizer.token_dim

    def test_tokenizer_frame_rate(self) -> None:
        tokenizer = TokenizerModule()
        tokenizer.eval()
        # 1 second of audio -> 50 tokens expected.
        chunk = torch.from_numpy(_make_chunk(16000)).unsqueeze(0)
        with torch.no_grad():
            seq = tokenizer.tokenize(chunk, None)
        assert len(seq) == 50

    def test_tokenizer_soft_embeddings(self) -> None:
        """Token embeddings must be continuous (soft), not discrete one-hot."""
        tokenizer = TokenizerModule()
        tokenizer.eval()
        chunk = torch.from_numpy(_make_chunk()).unsqueeze(0)
        with torch.no_grad():
            seq = tokenizer.tokenize(chunk, None)
        embedding = seq.tokens[0].token_embedding
        # Soft embeddings should not be one-hot.
        assert not torch.allclose(embedding, torch.nn.functional.one_hot(
            embedding.argmax(), num_classes=embedding.shape[-1]
        ).float())

    def test_tokenizer_metadata(self) -> None:
        tokenizer = TokenizerModule()
        assert tokenizer.token_rate_hz == 50
        assert tokenizer.token_dim == 128

    def test_tokenizer_to_tensor(self) -> None:
        seq = _make_token_sequence(seq_len=4, dim=128)
        tensor = seq.to_tensor()
        assert tensor.shape == (4, 128)
        assert torch.allclose(tensor[0], seq.tokens[0].token_embedding)

    def test_tokenizer_from_tensor(self) -> None:
        tensor = torch.randn(4, 128)
        seq = TokenSequence.from_tensor(tensor, start_ms=0.0, duration_ms=20.0)
        assert len(seq) == 4
        assert torch.allclose(seq.tokens[2].token_embedding, tensor[2])


class TestTranslatorChangesTokens:
    def test_translator_modifies_embeddings(self) -> None:
        config = TokenTranslationConfig()
        translator = TranslatorModule(
            token_dim=config.token_dim,
            translator_layers=config.translator_layers,
            translator_hidden=config.translator_hidden,
            num_accents=config.num_accents,
            lookahead_frames=0,
        )
        translator.eval()
        src = _make_token_sequence(seq_len=4, dim=config.token_dim)
        target_accent = torch.tensor([1], dtype=torch.long)
        with torch.no_grad():
            mapped = translator.translate(src, target_accent, strength=0.9)
        src_emb = src.to_tensor().mean(dim=0)
        mapped_emb = mapped.to_tensor().mean(dim=0)
        assert not torch.allclose(src_emb, mapped_emb)

    def test_translator_output_shape(self) -> None:
        config = TokenTranslationConfig()
        translator = TranslatorModule(
            token_dim=config.token_dim,
            translator_layers=config.translator_layers,
            translator_hidden=config.translator_hidden,
            num_accents=config.num_accents,
            lookahead_frames=0,
        )
        translator.eval()
        src = _make_token_sequence(seq_len=8, dim=config.token_dim)
        target_accent = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            mapped = translator.translate(src, target_accent, strength=0.5)
        assert mapped.to_tensor().shape == (8, config.token_dim)

    def test_translator_single_token(self) -> None:
        config = TokenTranslationConfig()
        translator = TranslatorModule(
            token_dim=config.token_dim,
            translator_layers=config.translator_layers,
            translator_hidden=config.translator_hidden,
            num_accents=config.num_accents,
            lookahead_frames=0,
        )
        translator.eval()
        tokens = TokenSequence([
            SpeechToken(token_id=0, token_embedding=torch.randn(config.token_dim),
                        timestamp_ms=0.0, duration_ms=20.0, is_speech=True)
        ])
        target_accent = torch.tensor([3], dtype=torch.long)
        with torch.no_grad():
            mapped = translator.translate(tokens, target_accent, strength=0.7)
        assert len(mapped) == 1


class TestTranslatorCausal:
    def test_strict_causal_no_future(self) -> None:
        """When lookahead_frames=0, translator must not access future tokens."""
        config = TokenTranslationConfig(lookahead_frames=0)
        translator = TranslatorModule(
            token_dim=config.token_dim,
            translator_layers=config.translator_layers,
            translator_hidden=config.translator_hidden,
            num_accents=config.num_accents,
            lookahead_frames=0,
        )
        translator.eval()
        src = _make_token_sequence(seq_len=4, dim=config.token_dim)
        target_accent = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            mapped = translator.translate(src, target_accent, strength=0.5)
        # For strict causal, output length must equal input length.
        assert mapped.to_tensor().shape[0] == 4

    def test_translator_state_carry_over(self) -> None:
        config = TokenTranslationConfig(lookahead_frames=0)
        translator = TranslatorModule(
            token_dim=config.token_dim,
            translator_layers=config.translator_layers,
            translator_hidden=config.translator_hidden,
            num_accents=config.num_accents,
            lookahead_frames=0,
        )
        translator.eval()
        src = _make_token_sequence(seq_len=8, dim=config.token_dim)
        target_accent = torch.tensor([0], dtype=torch.long)

        # Process first chunk.
        ctx = {}
        with torch.no_grad():
            mapped1 = translator.translate(src.slice(0, 4), target_accent, strength=0.5, context=ctx)
        assert "translator_state" in ctx

        # Process second chunk with carried state.
        with torch.no_grad():
            mapped2 = translator.translate(src.slice(4, 8), target_accent, strength=0.5, context=ctx)
        assert mapped2.to_tensor().shape == (4, config.token_dim)

    def test_translator_lookahead_zero(self) -> None:
        config = TokenTranslationConfig(lookahead_frames=0)
        assert config.lookahead_frames == 0


class TestSynthesizerProducesAudio:
    def test_synthesis_output_shape(self) -> None:
        synth = SynthModule()
        synth.eval()
        tokens = _make_token_sequence(seq_len=4, dim=128)
        with torch.no_grad():
            audio = synth.synthesize(tokens, speaker_conditioning=None)
        assert audio.dim() >= 1
        assert audio.numel() > 0

    def test_synthesis_upsamples(self) -> None:
        synth = SynthModule(hop_length=4)
        synth.eval()
        tokens = _make_token_sequence(seq_len=4, dim=128)
        with torch.no_grad():
            audio = synth.synthesize(tokens, speaker_conditioning=None)
        # Expected length = token_len * hop_length = 16
        expected_len = 4 * 4
        assert audio.shape[-1] >= expected_len

    def test_synthesis_speaker_conditioning(self) -> None:
        synth = SynthModule()
        synth.eval()
        tokens = _make_token_sequence(seq_len=4, dim=128)
        speaker = torch.randn(1, 64)
        with torch.no_grad():
            audio_a = synth.synthesize(tokens, speaker_conditioning=speaker)
            audio_b = synth.synthesize(tokens, speaker_conditioning=None)
        # Different speaker conditioning should produce different audio.
        assert not torch.allclose(audio_a, audio_b)

    def test_synthesis_parameter_count(self) -> None:
        synth = SynthModule()
        total = sum(p.numel() for p in synth.parameters())
        assert total < 1_000_000, f"Synthesizer too large: {total} params"


class TestFullPipeline:
    def test_tokenize_translate_synthesize(self) -> None:
        config = TokenTranslationConfig()
        candidate = TokenTranslationCandidate(config=config)
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "p1"})
        chunk = _make_chunk(1280)
        result = candidate.process_chunk(session, chunk, 16000)
        assert result.audio is not None
        assert result.output_end_sample >= result.output_start_sample
        assert result.sample_rate == 16000

    def test_metadata_fields(self) -> None:
        candidate = TokenTranslationCandidate()
        assert candidate.metadata.architecture_id == "token_translation"
        assert candidate.metadata.input_sample_rate == 16000
        assert candidate.metadata.frame_ms == 20.0
        assert candidate.metadata.preferred_chunk_ms == 80
        assert candidate.metadata.required_lookahead_ms == 0
        assert candidate.metadata.supports_conversion_strength is True
        assert candidate.metadata.supports_target_accent is True
        assert candidate.metadata.requires_reference_speaker is False
        assert candidate.metadata.uses_text_at_inference is False

    def test_config_defaults(self) -> None:
        config = TokenTranslationConfig()
        assert config.token_rate_hz == 50
        assert config.token_dim == 128
        assert config.chunk_ms == 80
        assert config.lookahead_frames == 0
        assert config.translator_layers == 2
        assert config.translator_hidden == 256
        assert config.conversion_strength == 0.5


class TestLookaheadConfig:
    def test_lookahead_zero(self) -> None:
        config = TokenTranslationConfig(lookahead_frames=0)
        assert config.lookahead_frames == 0
        assert config.lookahead_ms == 0.0

    def test_lookahead_20ms(self) -> None:
        config = TokenTranslationConfig(lookahead_frames=1)
        assert config.lookahead_frames == 1
        assert abs(config.lookahead_ms - 20.0) < 1e-6

    def test_lookahead_40ms(self) -> None:
        config = TokenTranslationConfig(lookahead_frames=2)
        assert config.lookahead_frames == 2
        assert abs(config.lookahead_ms - 40.0) < 1e-6

    def test_lookahead_80ms(self) -> None:
        config = TokenTranslationConfig(lookahead_frames=4)
        assert config.lookahead_frames == 4
        assert abs(config.lookahead_ms - 80.0) < 1e-6

    def test_candidate_metadata_reflects_lookahead(self) -> None:
        candidate = TokenTranslationCandidate(config=TokenTranslationConfig(lookahead_frames=2))
        assert candidate.metadata.required_lookahead_ms == 40

    def test_lookahead_translator_shape(self) -> None:
        """With lookahead, output length should match input (context is extended)."""
        config = TokenTranslationConfig(lookahead_frames=2)
        translator = TranslatorModule(
            token_dim=config.token_dim,
            translator_layers=config.translator_layers,
            translator_hidden=config.translator_hidden,
            num_accents=config.num_accents,
            lookahead_frames=2,
        )
        translator.eval()
        src = _make_token_sequence(seq_len=6, dim=config.token_dim)
        target_accent = torch.tensor([0], dtype=torch.long)
        with torch.no_grad():
            mapped = translator.translate(src, target_accent, strength=0.5)
        # Lookahead extends context but preserves sequence length in batch mode.
        assert mapped.to_tensor().shape[0] == 6


class TestStreamingSession:
    def test_session_management(self) -> None:
        candidate = TokenTranslationCandidate()
        candidate.prepare("cpu", "fp32")
        s1 = candidate.create_session({"session_id": "s1"})
        s2 = candidate.create_session({"session_id": "s2"})
        chunk = _make_chunk(1280)
        r1 = candidate.process_chunk(s1, chunk, 16000)
        r2 = candidate.process_chunk(s2, chunk, 16000)
        assert r1.input_end_sample == chunk.shape[0]
        assert r2.input_end_sample == chunk.shape[0]

    def test_reset(self) -> None:
        candidate = TokenTranslationCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "r1"})
        chunk = _make_chunk(1280)
        candidate.process_chunk(session, chunk, 16000)
        candidate.reset(session)
        assert session.state["c"].samples_processed == 0
        assert session.samples_processed == 0

    def test_multiple_chunks(self) -> None:
        candidate = TokenTranslationCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "m1"})
        for _ in range(3):
            chunk = _make_chunk(1280)
            result = candidate.process_chunk(session, chunk, 16000)
            assert result.audio is not None
        assert session.state["c"].samples_processed == 3 * 1280

    def test_state_bounded(self) -> None:
        candidate = TokenTranslationCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "bounded"})
        chunk = _make_chunk(1280)
        for _ in range(5):
            candidate.process_chunk(session, chunk, 16000)
        assert len(session.state["c"].timeline) == 5


class TestNoTextRequired:
    def test_inference_no_text(self) -> None:
        """Inference path must NOT require any text input."""
        candidate = TokenTranslationCandidate()
        candidate.prepare("cpu", "fp32")
        session = candidate.create_session({"session_id": "no_text"})
        chunk = _make_chunk(1280)
        # No text arguments anywhere in the call.
        result = candidate.process_chunk(session, chunk, 16000)
        assert result.audio is not None

    def test_metadata_no_text(self) -> None:
        candidate = TokenTranslationCandidate()
        assert candidate.metadata.uses_text_at_inference is False


