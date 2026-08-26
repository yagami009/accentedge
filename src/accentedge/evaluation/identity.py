"""Phase 1 — Speaker identity evaluation.

Uses SpeechBrain ECAPA-TDNN (spkrec-ecapa-voxceleb) for speaker embedding.

Key Phase-1 quantity:
  identity_drop(strength) = sim(source, reconstruction) - sim(source, converted)
"""
from __future__ import annotations

import numpy as np
import torch
import soundfile as sf


class IdentityEvaluator:
    """ECAPA-TDNN speaker similarity evaluator."""

    def __init__(self, device: str = "cpu"):
        from speechbrain.pretrained import EncoderClassifier
        self.classifier = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir="/tmp/spkrec_ecapa",
        )
        self.device = device

    @torch.no_grad()
    def embed(self, waveform: np.ndarray, sr: int = 24000) -> torch.Tensor:
        """Compute speaker embedding for a waveform."""
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        wav_t = torch.tensor(waveform, dtype=torch.float32).unsqueeze(0)
        emb = self.classifier.encode_batch(wav_t)
        return (emb / emb.norm(dim=-1, keepdim=True)).squeeze()

    def similarity(self, src: np.ndarray, tgt: np.ndarray, sr: int = 24000) -> float:
        """Cosine similarity between two waveforms."""
        emb_src = self.embed(src, sr)
        emb_tgt = self.embed(tgt, sr)
        return float(torch.dot(emb_src, emb_tgt).item())

    def identity_drop(self, source: np.ndarray, reconstruction: np.ndarray,
                      converted: np.ndarray, sr: int = 24000) -> dict:
        """Compute identity drop relative to reconstruction ceiling."""
        sim_recon = self.similarity(source, reconstruction, sr)
        sim_conv = self.similarity(source, converted, sr)
        return {
            "sim_reconstruction": sim_recon,
            "sim_converted": sim_conv,
            "identity_drop": sim_recon - sim_conv,
        }
