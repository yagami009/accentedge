"""Phase 1 — FACodec adapter (using Amphion's bundled FACodec).

Wraps Amphion's ns3_codec FACodecEncoder + FACodecDecoder as
AccentEdge's frozen codec backend.

Key verified facts:
  - Sample rate: 24000 Hz
  - Content: zc (2 codebooks, dim=8)
  - Prosody: zp (1 codebook, dim=8)
  - Acoustic detail: zr (3 codebooks, dim=8)
  - Timbre: speaker_embedding (dim=1024)
  - Frame rate: 50 fps (hop_length=240 at 24kHz)

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

from models.codec.ns3_codec.facodec import FACodecEncoder, FACodecDecoder
from huggingface_hub import hf_hub_download


class FACodecAdapter(FactorizedSpeechCodec):
    """FACodec codec adapter using Amphion's bundled implementation.

    Uses upstream FAcodec pattern: encoder -> quantizer -> decoder
    The quantizer returns z (combined), quantized_list, commitment_loss, codebook_loss, timbre.
    The decoder receives z directly (no manual summing).
    """

    sample_rate: int = 24000

    def __init__(self, device: str = "cpu", facodec_ckpt: str = "Plachta/FAcodec"):
        self.device = torch.device(device)
        self.facodec_ckpt = facodec_ckpt

        # Download checkpoint
        ckpt_path, config_path = hf_hub_download(
            repo_id=facodec_ckpt, filename="pytorch_model.bin"
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        model_params = config.get("model_params", config.get("model", {}))

        # Build encoder and decoder from Amphion
        encoder = FACodecEncoder(
            ngf=model_params.get("ngf", 32),
            up_ratios=tuple(model_params.get("up_ratios", [2, 4, 5, 5])),
            out_channels=model_params.get("out_channels", 1024),
        )
        decoder = FACodecDecoder(
            in_channels=model_params.get("decoder_in_channels", 256),
            upsample_initial_channel=model_params.get("decoder_upsample_initial_channel", 1536),
            ngf=model_params.get("decoder_ngf", 32),
            up_ratios=tuple(model_params.get("decoder_up_ratios", [5, 5, 4, 2])),
            vq_num_q_c=model_params.get("vq_num_q_c", 2),
            vq_num_q_p=model_params.get("vq_num_q_p", 1),
            vq_num_q_r=model_params.get("vq_num_q_r", 3),
            vq_dim=model_params.get("vq_dim", 256),
            codebook_dim=model_params.get("codebook_dim", 8),
            codebook_size_prosody=model_params.get("codebook_size_prosody", 1024),
            codebook_size_content=model_params.get("codebook_size_content", 1024),
            codebook_size_residual=model_params.get("codebook_size_residual", 1024),
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

        wav_24k = torchaudio.functional.resample(waveform, 16000, 24000)
        wav_in = wav_24k.unsqueeze(0).float().to(self.device)

        # Upstream FAcodec pattern: encoder -> quantizer
        # quantizer returns: z, quantized_list, commitment_loss, codebook_loss, timbre
        z = self.model["encoder"](wav_in)
        z, quantized_list, commitment_loss, codebook_loss, timbre = self.model["quantizer"](
            z, wav_in, n_c=2
        )
        # quantized_list = [z_c, z_p, z_r]
        z_c, z_p, z_r = quantized_list

        return FactorizedLatents(
            content=z_c,
            content_zc1=z_c,
            content_zc2=None,
            prosody=z_p,
            detail=z_r,
            timbre=timbre,
            metadata={"sample_rate": self.sample_rate, "ckpt_hash": self._ckpt_hash},
        )

    @torch.no_grad()
    def decode(self, latents: FactorizedLatents) -> torch.Tensor:
        # Upstream pattern: decoder receives z directly (timbre baked into z by quantizer)
        # Note: latents.content here is actually the full quantized z (content + prosody + residual)
        waveform = self.model["decoder"](latents.content.to(self.device))
        return waveform.cpu()

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
