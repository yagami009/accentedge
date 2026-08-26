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

from codec.interfaces import FactorizedLatents, FactorizedSpeechCodec


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

        Reconstructs from content + prosody + timbre.
        zc2 is zeroed (not available during conversion in Phase 1).
        """
        # Convert zc1 codes to continuous vectors
        z_c = self._codes_to_vector(latents.content_zc1, quantizer_idx=1, layer=0)  # [B, D, T]

        z = z_c
        if latents.prosody is not None:
            z_p = self._codes_to_vector(latents.prosody, quantizer_idx=0, layer=0)
            z = z + z_p

        # Timbre: add if available
        if latents.timbre is not None:
            # Timbre is a global vector [B, D]; expand to sequence
            # In the full codec, timbre conditions the decoder
            # For Phase 1, we add it as a mean offset
            timbre_mean = latents.timbre.mean(dim=-1, keepdim=True).unsqueeze(-1)  # [B, D, 1]
            z = z + timbre_mean

        with torch.no_grad():
            waveform = self.model.decoder(z.to(self.device))
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
