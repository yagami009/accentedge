#!/usr/bin/env python3
"""Phoneme conditioning pipeline for Phase 1.

.. deprecated::
    This module is deprecated. Use :mod:`accentedge.phase1.phoneme_pipeline`
    instead, which implements the full paper-faithful pipeline:

        transcript -> eSpeak-ng phonemizer -> phoneme sequence
                   -> Wav2Vec2-XLSR CTC forced alignment -> 80fps frame IDs

    Migration::

        # OLD (deprecated):
        from accentedge.evaluation.phonemes import PhonemeConditioner
        conditioner = PhonemeConditioner(device='cuda')
        logits = conditioner.extract_phonemes(waveform)  # wrong model, no phonemizer

        # NEW (recommended):
        from accentedge.phase1.phoneme_pipeline import PhonemePipeline
        pipeline = PhonemePipeline(device='cuda')
        phone_ids = pipeline(transcript, waveform)  # [1, T] at 80fps, paper-faithful

    Key differences from the deprecated PhonemeConditioner:
    - Uses eSpeak-ng phonemizer (as specified in the paper)
    - Aligns phones to audio using forced alignment, not just frame resampling
    - Outputs exact [1, T] tensors matching FACodec frame count
    - Uses a Wav2Vec2 CTC phone model for forced alignment (not ASR decoding)
"""
import warnings

import torch
import torch.nn.functional as F
from typing import Optional

# Try to import transformers
try:
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


class PhonemeConditioner:
    """Extracts phoneme conditioning from source audio.

    .. deprecated::
        Use :class:`accentedge.phase1.phoneme_pipeline.PhonemePipeline` instead.
        This class does not use the paper's eSpeak-ng phonemizer and does not
        produce transcript-aligned phoneme boundaries. It may produce incorrect
        results compared to the paper's method.

    Uses Wav2Vec2-XLSR phoneme recognizer to produce frame-level
    phone posteriors at the codec frame rate (80 fps for 24kHz audio, 12.5ms per frame).
    """

    def __init__(self, device: str = "cpu", sample_rate: int = 24000):
        warnings.warn(
            "PhonemeConditioner from accentedge.evaluation.phonemes is deprecated. "
            "Use accentedge.phase1.phoneme_pipeline.PhonemePipeline instead, "
            "which uses eSpeak-ng phonemizer and correct 80fps forced alignment.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.device = torch.device(device)
        self.sample_rate = sample_rate
        self.frame_rate = sample_rate // 300  # hop_length=300, 24000/300 = 80fps
        self._model = None
        self._processor = None

    def _load_model(self):
        """Lazy-load the phoneme recognition model."""
        if self._model is not None:
            return
        if not HAS_TRANSFORMERS:
            raise ImportError("transformers required for phoneme conditioning: pip install transformers")
        # Use Wav2Vec2-XLSR-base which has phoneme recognition capability
        # This model is pretrained on 20+ languages and can be fine-tuned for phonemes
        model_name = "facebook/wav2vec2-xlsr-base"
        self._processor = Wav2Vec2Processor.from_pretrained(model_name)
        self._model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self._model.to(self.device)
        self._model.eval()

    @torch.no_grad()
    def extract_phonemes(self, waveform: torch.Tensor) -> torch.Tensor:
        """Extract phoneme logits from waveform.

        .. deprecated::
            Use :meth:`accentedge.phase1.phoneme_pipeline.PhonemePipeline.__call__`
            instead for paper-faithful phoneme conditioning.

        Args:
            waveform: [1, T] float32 audio at self.sample_rate

        Returns:
            phone_logits: [1, N_frames, phone_vocab_size]
                          at codec frame rate (80 fps for 24kHz audio)
        """
        warnings.warn(
            "PhonemeConditioner.extract_phonemes is deprecated. "
            "Use PhonemePipeline (accentedge.phase1.phoneme_pipeline) instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self._load_model()

        # Resample if needed
        if waveform.shape[-1] != self.sample_rate:
            import torchaudio
            waveform = torchaudio.functional.resample(
                waveform, waveform.shape[-1], self.sample_rate
            )

        # Get CTC logits from Wav2Vec2
        inputs = self._processor(
            waveform.squeeze(0).cpu().numpy(),
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self._model(**inputs).logits  # [1, N_ctc_frames, vocab]

        # Resample to codec frame rate
        # Wav2Vec2 outputs at ~50Hz for 16kHz input, ~75Hz for 24kHz input
        # We need to match the codec frame rate (80fps)
        n_frames = logits.shape[1]
        target_frames = max(1, int(n_frames * self.frame_rate / 75))
        logits = F.interpolate(
            logits.transpose(1, 2),
            size=target_frames,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

        return logits

    def get_phone_embeddings(self, waveform: torch.Tensor, phone_embed_dim: int = 256) -> torch.Tensor:
        """Get phone embeddings for conditioning.

        .. deprecated::
            Use :meth:`accentedge.phase1.phoneme_pipeline.PhonemePipeline.__call__`
            and the denoiser's built-in phone embedding instead.

        Args:
            waveform: [1, T] float32 audio
            phone_embed_dim: dimension of output phone embeddings

        Returns:
            phone_emb: [1, N_frames, phone_embed_dim]
        """
        warnings.warn(
            "PhonemeConditioner.get_phone_embeddings is deprecated. "
            "Use PhonemePipeline + DenoisingTransformerModel instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        logits = self.extract_phonemes(waveform)
        # Softmax over phone vocab, then project to embedding dim
        phone_probs = F.softmax(logits, dim=-1)
        phone_emb = F.linear(
            phone_probs,
            torch.randn(phone_probs.shape[-1], phone_embed_dim, device=self.device),
        )
        return phone_emb

