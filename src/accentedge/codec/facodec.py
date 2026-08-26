"""Phase 1 — FACodec adapter (using Amphion's bundled FACodec).

Wraps Amphion's ns3_codec FACodecEncoder + FACodecDecoder as
AccentEdge's frozen codec backend.

Key verified facts:
  - Sample rate: 24000 Hz
  - Content: zc1 (2 codebooks, dim=8) + zc2 (predicted after denoising)
  - Prosody: zp (1 codebook, dim=8)
  - Acoustic detail: zr (3 codebooks, dim=8)
  - Timbre: spk_embs (single vector per utterance, dim=256)
  - Frame rate: 50 fps (hop_length=300 at 24kHz)

Upstream: https://github.com/open-mmlab/Amphion
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

# Add Amphion to path
_AMPHION_PATH = Path(__file__).resolve().parents[3] / ".." / "Amphion"
_AMPHION_PATH = _AMPHION_PATH.resolve()
if str(_AMPHION_PATH) not in sys.path:
    sys.path.insert(0, str(_AMPHION_PATH))

from accentedge.codec.interfaces import FactorizedLatents, FactorizedSpeechCodec


class FACodecAdapter(nn.Module, FactorizedSpeechCodec):
    """Wraps Amphion's FACodec as AccentEdge's frozen codec."""

    sample_rate: int = 24000

    def __init__(self, device: str = "cpu", facodec_ckpt: str = "Plachta/FAcodec"):
        super().__init__()
        self.device = torch.device(device)

        from models.codec.ns3_codec.facodec import FACodecEncoder, FACodecDecoder
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(repo_id=facodec_ckpt, filename="pytorch_model.bin")
        config_path = hf_hub_download(repo_id=facodec_ckpt, filename="config.yml")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_params = config.get("model_params", config.get("model", {}))

        # Build encoder
        encoder = FACodecEncoder(ngf=32, up_ratios=[2, 4, 5, 5], out_channels=256)
        decoder = FACodecDecoder(
            in_channels=256, upsample_initial_channel=1024, ngf=32,
            up_ratios=[5, 5, 4, 2], vq_num_q_c=2, vq_num_q_p=1, vq_num_q_r=3,
            vq_dim=256, codebook_dim=8,
            codebook_size_prosody=10, codebook_size_content=10, codebook_size_residual=10,
            use_gr_x_timbre=True,
        )

        self.model = {"encoder": encoder, "decoder": decoder}

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = ckpt.get("net", ckpt)
        if "encoder" in ckpt:
            self.model["encoder"].load_state_dict(ckpt["encoder"])
        if "decoder" in ckpt:
            self.model["decoder"].load_state_dict(ckpt["decoder"])
        for key in self.model:
            if key not in ckpt:
                for alt_key in ckpt:
                    if key.lower() in alt_key.lower():
                        self.model[key].load_state_dict(ckpt[alt_key])
                        break

        for key in self.model:
            self.model[key].eval().to(self.device)
        for key in self.model:
            for param in self.model[key].parameters():
                param.requires_grad = False

        self._ckpt_hash = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()[:16]

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor):
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        z = self.model["encoder"](waveform[None, ...].to(self.device).float())
        _, quantized, _, _, timbre, codes = self.model["decoder"](
            z, waveform[None, ...].to(self.device).float(), return_codes=True, n_c=2
        )

        codes_c, codes_p, codes_t, codes_r = codes
        z_c, z_p, z_t, z_r = quantized

        return FactorizedLatents(
            content=z_c,
            content_zc1=codes_c[0] if isinstance(codes_c, (list, tuple)) else codes_c,
            content_zc2=codes_c[1] if isinstance(codes_c, (list, tuple)) and len(codes_c) > 1 else None,
            prosody=codes_p,
            detail=codes_r,
            timbre=timbre,
            metadata={"sample_rate": self.sample_rate, "ckpt_hash": self._ckpt_hash},
        )

    @torch.no_grad()
    def decode(self, latents: FactorizedLatents) -> torch.Tensor:
        # Reconstruct all factors
        z_c1 = self._codes_to_vector(latents.content_zc1, quantizer_idx=1, layer=0)
        z_c = z_c1
        if latents.content_zc2 is not None:
            z_c2 = self._codes_to_vector(latents.content_zc2, quantizer_idx=1, layer=1)
            z_c = z_c + z_c2

        z_p = 0.0
        if latents.prosody is not None:
            z_p = self._codes_to_vector(latents.prosody, quantizer_idx=0, layer=0)

        z_r = 0.0
        if latents.detail is not None:
            for k in range(latents.detail.shape[1]):
                z_r += self._codes_to_vector(latents.detail[:, k:k+1, :], quantizer_idx=2, layer=k)

        z = z_c + z_p + z_r

        speaker_embedding = latents.timbre if latents.timbre is not None else torch.zeros(
            z.shape[0], 256, device=z.device, dtype=z.dtype
        )

        waveform = self.model["decoder"].inference(
            z.to(self.device), speaker_embedding=speaker_embedding.to(self.device)
        )
        return waveform.cpu()

    def _codes_to_vector(self, code_indices, quantizer_idx: int = 0, layer: int = 0) -> torch.Tensor:
        codebook = self.model["decoder"].quantizer[quantizer_idx].layers[layer].codebook.weight
        z = torch.nn.functional.embedding(code_indices.squeeze(0), codebook)
        z = z.transpose(1, 2).unsqueeze(0)
        if hasattr(self.model["decoder"].quantizer[quantizer_idx].layers[layer], "out_proj"):
            z = self.model["decoder"].quantizer[quantizer_idx].layers[layer].out_proj(z)
        return z

    def freeze(self) -> None:
        frozen = True
        for key in self.model:
            for param in self.model[key].parameters():
                if param.requires_grad:
                    frozen = False
        assert frozen, "FACodec parameters must be frozen"
        print("FACodec: all parameters frozen")

    @torch.no_grad()
    def reconstruction(self, waveform: torch.Tensor) -> torch.Tensor:
        latents = self.encode(waveform)
        return self.decode(latents)
