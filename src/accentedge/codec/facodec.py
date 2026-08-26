"""Phase 1 — FACodec adapter (using standalone Plachta/FAcodec).

Wraps Plachta/FAcodec (NaturalSpeech 3 factorized codec) as AccentEdge's
frozen codec backend. The codec is frozen after initialization.

Key verified facts:
  - Sample rate: 24000 Hz
  - Content: zc1 (2 codebooks, dim=8) + zc2 (predicted after denoising)
  - Prosody: zp (1 codebook, dim=8)
  - Acoustic detail: zr (3 codebooks, dim=8) — residual after content+prosody+timbre
  - Timbre: g (single vector per utterance, dim=1024)
  - Frame rate: 50 fps (hop_length=300 at 24kHz)
  - Codebook: 1024 entries, 8-dim each

Upstream: https://github.com/Plachtaa/FAcodec
Checkpoint: Plachta/FAcodec (HuggingFace)
"""
from __future__ import annotations

import os
import sys
import hashlib
import warnings
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import numpy as np
import soundfile as sf
import yaml

warnings.simplefilter("ignore")

# Path to standalone FAcodec repo (verified working)
_FACODEC_REPO = Path(__file__).resolve().parents[3] / ".." / "FAcodec"
_FACODEC_REPO = _FACODEC_REPO.resolve()

if str(_FACODEC_REPO) not in sys.path:
    sys.path.insert(0, str(_FACODEC_REPO))

from accentedge.codec.interfaces import FactorizedLatents, FactorizedSpeechCodec


class FACodecAdapter(nn.Module, FactorizedSpeechCodec):
    """Wraps Plachta/FAcodec as AccentEdge's frozen codec.

    Usage:
        adapter = FACodecAdapter(device="cpu")
        latents = adapter.encode(waveform)
        recon = adapter.decode(latents)
    """

    sample_rate: int = 24000

    def __init__(self, device: str = "cpu", facodec_ckpt: str = "Plachta/FAcodec"):
        super().__init__()
        self.device = torch.device(device)

        # Import FAcodec modules
        from modules.commons import build_model, recursive_munch
        from hf_utils import load_custom_model_from_hf

        ckpt_path, config_path = load_custom_model_from_hf(facodec_ckpt)
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_params = recursive_munch(config.get("model_params", config.get("model", {})))
        self.model = build_model(model_params)

        ckpt_params = torch.load(ckpt_path, map_location="cpu")
        for key in ckpt_params:
            self.model[key].load_state_dict(ckpt_params[key])

        # Move all sub-models to device
        for key in self.model:
            self.model[key].eval().to(self.device)

        # Freeze all parameters
        if True:  # freeze_codec flag
            for key in self.model:
                for param in self.model[key].parameters():
                    param.requires_grad = False

        # Cache config
        self._config_hash = hashlib.sha256(
            yaml.dump(model_params).encode()
        ).hexdigest()[:16]

        self._ckpt_hash = hashlib.sha256(
            open(ckpt_path, "rb").read()
        ).hexdigest()[:16]

    # ──────────────────────────────────────────────────────────────────────────
    # Encode / decode
    # ──────────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor) -> FactorizedLatents:
        """Encode waveform → factorized latents.

        Args:
            waveform: [B, T] float32, range [-1, 1], sample_rate=24000

        Returns:
            FactorizedLatents with content (zc1 continuous), prosody, timbre
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        z = self.model.encoder(waveform[None, ...].to(self.device).float())
        _, quantized, _, _, timbre, codes = self.model.quantizer(
            z, waveform[None, ...].to(self.device).float(), return_codes=True, n_c=2
        )

        # codes: [codes_c, codes_p, codes_t, codes_r]
        # quantized: [z_c, z_p, z_t, z_r]
        codes_c, codes_p, codes_t, codes_r = codes
        z_c, z_p, z_t, z_r = quantized

        # z_c is [z_c1 + z_c2]; we need z_c1 separately
        # The quantizer returns the combined content; split into codebooks
        # codes_c[0] = zc1 indices, codes_c[1] = zc2 indices
        return FactorizedLatents(
            content=z_c,  # combined content [B, D, T]
            content_zc1=codes_c[0] if isinstance(codes_c, (list, tuple)) else codes_c,
            content_zc2=codes_c[1] if isinstance(codes_c, (list, tuple)) and len(codes_c) > 1 else None,
            prosody=codes_p,
            detail=codes_r,
            timbre=timbre,
            metadata={
                "sample_rate": self.sample_rate,
                "codec_hash": self._ckpt_hash,
                "config_hash": self._config_hash,
                "source_shape": list(waveform.shape),
            },
        )

    @torch.no_grad()
    def decode(self, latents: FactorizedLatents) -> torch.Tensor:
        """Decode factorized latents → waveform.

        Reconstructs from content + prosody + residual + timbre,
        matching the upstream FACodec reconstruction formula:
          z = z_p + z_c + z_r
          speaker_embedding → timbre_linear → FiLM conditioning
          waveform = decoder.inference(z, speaker_embedding)
        """
        # Reconstruct content from zc1 (and zc2 if available)
        z_c1 = self._codes_to_vector(latents.content_zc1, quantizer_idx=1, layer=0)
        z_c = z_c1
        if latents.content_zc2 is not None:
            z_c2 = self._codes_to_vector(latents.content_zc2, quantizer_idx=1, layer=1)
            z_c = z_c + z_c2

        # Reconstruct prosody
        if latents.prosody is not None:
            z_p = self._codes_to_vector(latents.prosody, quantizer_idx=0, layer=0)
        else:
            z_p = 0.0

        # Reconstruct residual
        if latents.detail is not None:
            # detail is [B, K, T] with K=3 residual quantizers
            z_r = 0.0
            for k in range(latents.detail.shape[1]):
                z_r += self._codes_to_vector(latents.detail[:, k:k+1, :], quantizer_idx=2, layer=k)
        else:
            z_r = 0.0

        # Sum all factors → [B, D, T]
        z = z_c + z_p + z_r

        # Timbre: pass as speaker_embedding to decoder's FiLM conditioning
        # This matches upstream: timbre_linear(spk_emb) → gamma, beta → FiLM
        # NOT a mean offset on the latent
        speaker_embedding = latents.timbre if latents.timbre is not None else torch.zeros(
            z.shape[0], 256, device=z.device, dtype=z.dtype
        )

        with torch.no_grad():
            waveform = self.model.inference(z.to(self.device), speaker_embedding=speaker_embedding.to(self.device))
        return waveform.cpu()

    def _codes_to_vector(self, code_indices, quantizer_idx: int = 0, layer: int = 0) -> torch.Tensor:
        """Convert codebook indices to continuous latent vectors."""
        codebook = self.model.quantizer[quantizer_idx].layers[layer].codebook.weight
        z = torch.nn.functional.embedding(code_indices.squeeze(0), codebook)
        z = z.transpose(1, 2).unsqueeze(0)  # [B, D, T]
        # Apply output projection if present
        if hasattr(self.model.quantizer[quantizer_idx].layers[layer], 'out_proj'):
            z = self.model.quantizer[quantizer_idx].layers[layer].out_proj(z)
        return z

    @torch.no_grad()
    def reconstruction(self, waveform: torch.Tensor) -> torch.Tensor:
        """Baseline: encode → decode without modification."""
        latents = self.encode(waveform)
        return self.decode(latents)

    def freeze(self) -> None:
        """Assert all codec parameters are frozen."""
        frozen = True
        for key in self.model:
            for param in self.model[key].parameters():
                if param.requires_grad:
                    frozen = False
                    print(f"WARNING: {key} parameter still requires_grad!")
        assert frozen, "FACodec parameters must be frozen in Phase 1"
        print("FACodec: all parameters frozen ✓")
