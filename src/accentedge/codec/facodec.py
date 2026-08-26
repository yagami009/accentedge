"""Phase 1 — FACodec adapter.

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

# Add Amphion path for FACodec models
_AMPHION_PATH = os.environ.get("AMPHION_PATH", str(Path(__file__).resolve().parents[5] / "Amphion"))
if os.path.exists(_AMPHION_PATH):
    sys.path.append(_AMPHION_PATH)

from codec.interfaces import FactorizedLatents, FactorizedSpeechCodec


def _load_facodec_from_hf(repo_id: str = "Plachta/FAcodec"):
    """Load FACodec encoder, quantizer, decoder from HuggingFace Hub."""
    from huggingface_hub import hf_hub_download

    ckpt_path = hf_hub_download(repo_id=repo_id, filename="facodec.pt")
    config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    model_params = config.get("model_params", config.get("model", {}))

    # Import Amphion FACodec
    try:
        from models.codec.ns3_codec import FACodecEncoder, FACodecDecoder
    except ImportError:
        raise ImportError(
            "Cannot import FACodecEncoder/FACodecDecoder from Amphion. "
            "Clone Amphion: git clone https://github.com/open-mmlab/Amphion.git"
        )

    encoder = FACodecEncoder(**model_params.get("DAC", {}))
    decoder = FACodecDecoder(**model_params.get("DAC", {}))

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if "net" in ckpt:
        ckpt = ckpt["net"]

    encoder.load_state_dict(ckpt.get("encoder", {}), strict=False)
    decoder.load_state_dict(ckpt.get("decoder", {}), strict=False)
    quantizer_state = ckpt.get("quantizer", {})

    return encoder, decoder, quantizer_state, model_params


class FACodecAdapter(FactorizedSpeechCodec):
    """AccentEdge wrapper for Plachta/FAcodec.

    Frozen by default. Provides FactorizedLatents with all factors.
    """

    sample_rate = 24000

    def __init__(self, device: str = "cpu"):
        self.device = torch.device(device)
        encoder, decoder, quantizer_state, model_params = _load_facodec_from_hf()
        self.encoder = encoder.to(self.device).eval()
        self.decoder = decoder.to(self.device).eval()
        self._quantizer_state = quantizer_state

        # Freeze all parameters
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.decoder.parameters():
            param.requires_grad = False

        # Model dimensions from config
        self.dim = model_params.get("DAC", {}).get("encoder_dim", 64)
        self._verify_frozen()

    def _verify_frozen(self):
        for name, param in self.encoder.named_parameters():
            assert not param.requires_grad, f"Encoder parameter {name} is trainable!"
        for name, param in self.decoder.named_parameters():
            assert not param.requires_grad, f"Decoder parameter {name} is trainable!"

    def encode(self, waveform: torch.Tensor) -> FactorizedLatents:
        """Encode waveform → factorized latents.

        Args:
            waveform: [B, T] float32, mono, 24kHz

        Returns:
            FactorizedLatents with all factors extracted
        """
        assert waveform.dim() == 2, f"Expected [B, T], got {waveform.shape}"
        waveform = waveform.to(self.device)

        with torch.no_grad():
            z = self.encoder(waveform)
            z_out, quantized, commitment_loss, codebook_loss, timbre, codes = \
                self.decoder.quantizer(z, waveform, n_c=2, return_codes=True)

        # codes[0] = prosody indices [B, 1, T]
        # codes[1] = content indices [B, 2, T] (2 codebooks for zc1)
        # quantized[0] = z_p [B, D, T]
        # quantized[1] = z_c [B, D, T]
        # quantized[2] = z_t [B, D, T]
        # quantized[3] = z_r [B, D, T]

        # Extract content codebook indices
        content_codes = codes[1]  # [B, 2, T] — zc1 and zc2 code indices
        prosody_codes = codes[0]  # [B, 1, T]
        # Timbre is already a vector [B, D]

        return FactorizedLatents(
            content=quantized[1],       # z_c [B, D, T] — content quantized
            content_zc1=content_codes[:, 0:1, :],  # zc1 code indices [B, 1, T]
            content_zc2=content_codes[:, 1:2, :],  # zc2 code indices [B, 1, T]
            prosody=prosody_codes,       # [B, 1, T]
            detail=quantized[3],         # z_r [B, D, T] — residual detail
            timbre=timbre,               # [B, D] — global timbre
            metadata={
                "codes_prosody": codes[0].cpu(),
                "codes_content": codes[1].cpu(),
                "codes_acoustic": codes[2].cpu() if len(codes) > 2 else None,
            }
        )

    def decode(self, latents: FactorizedLatents) -> torch.Tensor:
        """Decode factorized latents → waveform.

        Reconstructs the quantized sum from available factors,
        then passes through the decoder.
        """
        # Reconstruct the quantized z that the decoder expects
        z = latents.content  # [B, D, T]

        if latents.prosody is not None:
            # Need to get prosody quantized vector from codebook indices
            prosody_indices = latents.prosody  # [B, 1, T]
            z_p = self._codes_to_vector(prosody_indices, quantizer_idx=0, layer=0)
            z = z + z_p

        if latents.detail is not None:
            z_r = latents.detail  # [B, D, T]
            z = z + z_r

        with torch.no_grad():
            waveform = self.decoder(z.to(self.device))
        return waveform.cpu()

    def _codes_to_vector(self, code_indices: torch.Tensor, quantizer_idx: int, layer: int = 0) -> torch.Tensor:
        """Convert codebook indices to continuous vectors via the codebook."""
        codebook = self.decoder.quantizer[quantizer_idx].layers[layer].codebook.weight
        # code_indices: [B, K, T] → [B, T, K]
        indices = code_indices.squeeze(1).long() if code_indices.dim() == 3 else code_indices.long()
        vectors = F.embedding(indices, codebook)  # [B, T, D]
        # Transpose to [B, D, T]
        return vectors.transpose(1, 2)

    def decode_from_codes(self, content_codes: torch.Tensor,
                          prosody_codes: Optional[torch.Tensor] = None,
                          timbre: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Decode from raw codebook indices (for denoised zc1).

        Args:
            content_codes: [B, 1, T] zc1 codebook indices
            prosody_codes: [B, 1, T] prosody codebook indices (source, preserved)
            timbre: [B, D] global timbre (source, preserved)
        """
        # Convert zc1 codes → continuous vectors
        z_c1 = self._codes_to_vector(content_codes, quantizer_idx=1, layer=0)  # [B, D, T]

        # zc2 stays zeroed (will be predicted or use source)
        z_c = z_c1  # [B, D, T]

        z = z_c
        if prosody_codes is not None:
            z_p = self._codes_to_vector(prosody_codes, quantizer_idx=0, layer=0)
            z = z + z_p

        # Timbre: if not provided, use zero (unconditional)
        # In Phase 1, timbre is always preserved from source

        with torch.no_grad():
            waveform = self.decoder(z.to(self.device))
        return waveform.cpu()

    def reconstruction(self, waveform: torch.Tensor) -> torch.Tensor:
        """Baseline: encode → decode without modification."""
        latents = self.encode(waveform)
        return self.decode(latents)
