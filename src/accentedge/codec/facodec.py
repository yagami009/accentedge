"""Phase 1 — FACodec adapter (using upstream FAcodec directly).

Wraps Plachta/FAcodec as AccentEdge's frozen codec backend.
Uses the exact upstream FAcodec pattern from reconstruct.py:
  1. encoder(wav) -> z
  2. quantizer(z, wav, n_c=2) -> z_q, [z_c, z_p, z_r], commitment_loss, codebook_loss, timbre
  3. decoder(z_q) -> waveform

Key verified facts:
  - Sample rate: 24000 Hz
  - Content: zc (2 codebooks, dim=8)
  - Prosody: zp (1 codebook, dim=8)
  - Acoustic detail: zr (3 codebooks, dim=8)
  - Timbre: speaker_embedding (dim=1024)
  - Frame rate: 50 fps (hop_length=240 at 24kHz)

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

from accentedge.codec.interfaces import FactorizedLatents, FactorizedSpeechCodec


class FACodecAdapter(FactorizedSpeechCodec):
    """FACodec codec adapter using upstream FAcodec directly.

    Uses the exact upstream FAcodec pattern from reconstruct.py.
    """

    sample_rate: int = 24000

    def __init__(self, device: str = "cpu", facodec_ckpt: str = "Plachta/FAcodec"):
        self.device = torch.device(device)
        self.facodec_ckpt = facodec_ckpt

        # Setup FAcodec path (FAC-FACodec pattern)
        _facodec_path = Path(__file__).resolve().parents[4] / "FAcodec"
        if _facodec_path.exists() and str(_facodec_path) not in sys.path:
            sys.path.insert(0, str(_facodec_path))

        _facodec_modules = _facodec_path / "modules"
        if _facodec_modules.exists() and str(_facodec_modules) not in sys.path:
            sys.path.insert(0, str(_facodec_modules))

        from modules.commons import build_model, recursive_munch
        from hf_utils import load_custom_model_from_hf

        ckpt_path, config_path = load_custom_model_from_hf(facodec_ckpt)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        model_params = recursive_munch(config["model_params"])
        self.model = build_model(model_params)

        ckpt = torch.load(ckpt_path, map_location="cpu")
        ckpt = ckpt.get("net", ckpt)
        for key in ckpt:
            self.model[key].load_state_dict(ckpt[key])
            self.model[key].eval()

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

        # Resample to 24kHz if needed
        wav = waveform.float()
        if wav.shape[-1] > 100000:  # likely 16kHz -> resample up
            wav = torchaudio.functional.resample(wav, 16000, 24000)
        wav_in = wav.unsqueeze(0).to(self.device)

        # Upstream FAcodec pattern: encoder -> quantizer
        # quantizer returns: z_q, [z_c, z_p, z_r], commitment_loss, codebook_loss, timbre
        z = self.model["encoder"](wav_in)
        z_q, quantized_list, commitment_loss, codebook_loss, timbre = self.model["quantizer"](
            z, wav_in, n_c=2
        )
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
        # Upstream pattern: decoder receives z_q directly (timbre baked into z by quantizer)
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
