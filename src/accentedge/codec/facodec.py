"""Phase 1 — FACodec adapter (using upstream FAcodec directly).

Wraps Plachta/FAcodec as AccentEdge's frozen codec backend.

Uses the exact upstream FAcodec pattern from reconstruct.py:
  1. encoder(wav) -> z
  2. quantizer(z, wav, n_c=2) -> z_q, [z_p, z_c, z_r], commitment_loss, codebook_loss, timbre
  3. decoder(z_q) -> waveform

For timbre_norm=True (default, as used by Plachta/FAcodec):
  - quantizer.forward_v2 applies StyleEncoder -> gamma/beta -> LayerNorm modulation
    directly on the summed latent. Timbre is baked into z_q before it is returned.
  - The quantized list is [z_p, z_c, z_r] (no separate z_t).
  - timbre is the raw StyleEncoder output [B, D].

For reconstruction: pass z_q directly to decoder.
For accent conversion: modify z_c and combine with z_p + z_r + timbre-conditioned z.
"""
from __future__ import annotations

import sys
import hashlib
import warnings
from pathlib import Path

import torch
import numpy as np

warnings.simplefilter("ignore")

from accentedge.codec.interfaces import FactorizedLatents, FactorizedSpeechCodec


class FACodecAdapter(FactorizedSpeechCodec):
    """FACodec codec adapter using upstream FAcodec directly."""

    sample_rate: int = 24000

    def __init__(self, device: str = "cpu", facodec_ckpt: str = "Plachta/FAcodec"):
        self.device = torch.device(device)
        self.facodec_ckpt = facodec_ckpt

        # Setup FAcodec path so `from modules.commons import ...` resolves
        _facodec_path = Path(__file__).resolve().parents[4] / "FAcodec"
        for _path in [_facodec_path, _facodec_path / "modules"]:
            if _path.exists() and str(_path) not in sys.path:
                sys.path.insert(0, str(_path))

        from modules.commons import build_model, recursive_munch
        from hf_utils import load_custom_model_from_hf

        try:
            ckpt_path, config_path = load_custom_model_from_hf(facodec_ckpt)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to download FAcodec checkpoint '{facodec_ckpt}'. "
                f"HuggingFace hub error: {exc}"
            ) from exc

        try:
            import yaml
            with open(config_path) as f:
                config = yaml.safe_load(f)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FAcodec config at '{config_path}': {exc}"
            ) from exc

        mp = recursive_munch(config["model_params"])
        try:
            self.model = build_model(mp)
        except Exception as exc:
            raise RuntimeError(f"Failed to build FAcodec model: {exc}") from exc

        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
            ckpt = ckpt.get("net", ckpt)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load FAcodec checkpoint from '{ckpt_path}': {exc}"
            ) from exc

        for key in ckpt:
            if key not in self.model:
                raise KeyError(
                    f"Checkpoint key '{key}' not found in model keys: {list(self.model.keys())}"
                )
            try:
                self.model[key].load_state_dict(ckpt[key])
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to load state dict for model key '{key}': {exc}"
                ) from exc

        for key in self.model:
            self.model[key].eval().to(self.device)
            for p in self.model[key].parameters():
                p.requires_grad = False

        # Determine quantizer mode from config for informative logging
        self._timbre_norm = getattr(mp, "timbre_norm", False)
        self._ckpt_hash = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()[:16]

    @torch.no_grad()
    def encode(self, waveform: torch.Tensor):
        """
        Encode waveform → FactorizedLatents.

        Upstream pattern (timbre_norm=True, the default):
          quantizer returns: z_q, [z_p, z_c, z_r], losses, timbre
          - z_q has timbre already baked in (StyleEncoder → gamma/beta → LayerNorm)
          - quantized list has 3 elements: [z_p, z_c, z_r]
          - timbre is the raw StyleEncoder output [B, D]

        timbre_norm=False:
          quantizer returns: z_q, [z_c, z_p, z_t, z_r], losses, timbre
          - quantized list has 4 elements: [z_c, z_p, z_t, z_r]
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        wav_in = waveform.unsqueeze(0).float().to(self.device)

        # Upstream FAcodec pattern: encoder -> quantizer -> decoder
        z = self.model["encoder"](wav_in)

        if self._timbre_norm:
            # forward_v2: returns (z_q, quantized=[z_p, z_c, z_r], losses, timbre)
            # z_q already has timbre conditioning baked in via LayerNorm modulation.
            z_q, quantized_list, commitment_loss, codebook_loss, timbre = self.model["quantizer"](
                z, wav_in, n_c=2
            )
            z_p, z_c, z_r = quantized_list
            # No separate z_t in timbre_norm mode; timbre is in z_q directly.
            z_t = None
        else:
            # forward: returns (z_q, quantized=[z_c, z_p, z_t, z_r], losses, timbre)
            z_q, quantized_list, commitment_loss, codebook_loss, timbre = self.model["quantizer"](
                z, wav_in, n_c=2
            )
            z_c, z_p, z_t, z_r = quantized_list

        # z_q is the timbre-conditioned quantized latent — this is exactly what the
        # decoder expects for faithful round-trip reconstruction.
        return FactorizedLatents(
            content=z_q,           # [B, C, T] — timbre-conditioned z for decoder input
            content_zc1=z_c,      # [B, 1, T] — content codebook indices
            content_zc2=None,      # not exposed by the upstream quantizer
            prosody=z_p,           # [B, 1, T] — prosody codebook indices
            detail=z_r,            # [B, K, T] — residual detail codebook indices
            timbre=timbre,        # [B, D] — raw StyleEncoder output (gamma/beta source)
            metadata={
                "sample_rate": self.sample_rate,
                "ckpt_hash": self._ckpt_hash,
                "timbre_norm": self._timbre_norm,
                # expose raw factors for accent conversion downstream
                "z_t": z_t,        # None when timbre_norm=True
                "z_c": z_c,
                "z_p": z_p,
                "z_r": z_r,
                "commitment_loss": commitment_loss,
                "codebook_loss": codebook_loss,
            },
        )

    @torch.no_grad()
    def decode(self, latents: FactorizedLatents) -> torch.Tensor:
        """
        Reconstruct waveform from FactorizedLatents.

        Reconstruction (encode → decode round-trip):
          Pass latents.content (z_q, timbre-conditioned) directly to decoder.
          This is the exact upstream pattern: decoder(z_q).

        Accent conversion (future):
          - Extract z_c_target from target-accent reference audio
          - Combine: z_modified = z_c_target + z_p + z_r
          - Timbre (gamma/beta) should come from source or target — TBD per strategy
          - Pass z_modified to decoder
        """
        z_q = latents.content.to(self.device)
        waveform = self.model["decoder"](z_q)
        return waveform.cpu()

    def freeze(self) -> None:
        frozen = True
        for key in self.model:
            for param in self.model[key].parameters():
                if param.requires_grad:
                    frozen = False
                    break
            if not frozen:
                break
        assert frozen, (
            "FACodec parameters must be frozen. "
            "Ensure no parameters have requires_grad=True after initialization."
        )
        print(f"FACodec: all parameters frozen (checkpoint hash: {self._ckpt_hash})")

    @torch.no_grad()
    def reconstruction(self, waveform: torch.Tensor) -> torch.Tensor:
        """Encode then decode for round-trip reconstruction."""
        latents = self.encode(waveform)
        return self.decode(latents)
