"""Phase 1 -- Accent conversion pipeline.

Orchestrates the full accent conversion flow:
  source_wav [1, T] at 24kHz
      -> FACodecAdapter.encode() -> z_q, z_c1, z_p, z_r, g
      -> PhonemePipeline(transcript, wav) -> phone_ids [1, T_frames] at 80fps
      -> Denoiser.denoise(z_q, phone_ids, strength) -> z_q_denoised, zc2_pred
      -> ZC2Recomputer.recompute(z_q_denoised, ...) -> valid zc2
      -> FACodecAdapter.decode(z_q_denoised) -> output_wav

Key design:
  - z_p, z_r, g are preserved from the source (prosody/timbre/detail)
  - Only z_q (the full quantized content) and zc2 are modified
  - The denoiser operates on the full 8-dim z_q representation, NOT on the
    1-dim z_c1. z_c1 is a separate codebook index stream that the denoiser
    predicts indirectly via the zc2 residual head.
"""
from __future__ import annotations

import torch

from accentedge.phase1.zc2_recompute import ZC2Recomputer


class AccentConverter:
    """Complete accent conversion pipeline.

    Pipeline:
        source_wav [1, T] at 24kHz
            |
            v
        FACodecAdapter.encode(wav) -> FactorizedLatents
            z_q [B, C, T]  -- full quantized content (timbre-conditioned)
            z_c1 [B, 1, T] -- content codebook indices
            z_p [B, 1, T]  -- prosody codebook indices
            z_r [B, K, T]  -- detail/residual codebook indices
            g [B, D]       -- timbre embedding
            |
            v
        PhonemePipeline(transcript, wav) -> phone_ids [1, T] at 80fps
            |
            v
        Denoiser(z_q, phone_ids, t) -> eps_pred, zc2_pred
            z_q_denoised = (z_q - sqrt(1mabar[t]) * eps) / sqrt(abar[t])
            |
            v
        z_q_new = interpolate(z_q, z_q_denoised, strength)
        zc2 = recompute(zc2_pred)
            |
            v
        FACodecAdapter.decode(z_q_new) -> output_wav [1, T]

    Args:
        facodec_adapter: FACodecAdapter instance (frozen)
        phoneme_pipeline: PhonemePipeline instance
        denoiser: DenoisingTransformerModel instance
        zc2_recomputer: ZC2Recomputer instance
        device: torch device
    """

    def __init__(
        self,
        facodec_adapter,
        phoneme_pipeline,
        denoiser,
        zc2_recomputer: ZC2Recomputer,
        device: torch.device,
    ):
        self.facodec_adapter = facodec_adapter
        self.phoneme_pipeline = phoneme_pipeline
        self.denoiser = denoiser
        self.zc2_recomputer = zc2_recomputer
        self.device = device

        # Validate components
        assert hasattr(self.facodec_adapter, "encode"), \
            "facodec_adapter must have encode()"
        assert hasattr(self.facodec_adapter, "decode"), \
            "facodec_adapter must have decode()"
        assert hasattr(self.phoneme_pipeline, "__call__"), \
            "phoneme_pipeline must be callable"
        assert hasattr(self.denoiser, "forward"), \
            "denoiser must have forward()"
        assert hasattr(self.zc2_recomputer, "recompute"), \
            "zc2_recomputer must have recompute()"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def convert(
        self,
        wav: torch.Tensor,
        transcript: str,
        strength: float = 1.0,
    ) -> torch.Tensor:
        """Run the full accent conversion pipeline.

        Args:
            wav: [1, T] float32 waveform at 24kHz.
            transcript: Source text transcript.
            strength: Conversion strength in [0.0, 1.0].
                      0.0 = no change, 1.0 = full denoising.

        Returns:
            output_wav: [1, T] float32 waveform at 24kHz.
        """
        result = self.convert_with_intermediates(wav, transcript, strength=strength)
        return result["output_wav"]

    def convert_with_intermediates(
        self,
        wav: torch.Tensor,
        transcript: str,
        strength: float = 1.0,
    ) -> dict:
        """Run the full pipeline and return all intermediate representations.

        Returns dict with keys:
            z_q, z_c1_original, z_p, z_r, g
            phone_ids
            z_q_denoised
            zc2_pred
            output_wav
        """
        # ----------------------------------------------------------------
        # Step 1: Validate input
        # ----------------------------------------------------------------
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if wav.dim() != 2 or wav.shape[0] != 1:
            raise ValueError(
                f"wav must be [1, T] or [T], got shape {tuple(wav.shape)}"
            )

        strength = max(0.0, min(1.0, strength))

        # ----------------------------------------------------------------
        # Step 2: Encode with FACodec
        # ----------------------------------------------------------------
        wav_device = wav.to(self.device)
        latents = self.facodec_adapter.encode(wav_device)

        z_q = latents.content
        z_c1_original = latents.content_zc1
        z_p = latents.prosody
        z_r = latents.detail
        g = latents.timbre

        self._validate_latent_shapes(z_q, z_c1_original, z_p, z_r, g)

        # ----------------------------------------------------------------
        # Step 3: Phoneme conditioning
        # ----------------------------------------------------------------
        phone_ids = self.phoneme_pipeline(transcript, wav_device)
        phone_ids = phone_ids.to(self.device)

        # Frame rate contract: phone_ids length must match z_q frame count
        self._assert_frame_rate_contract(phone_ids, z_q)

        # ----------------------------------------------------------------
        # Step 4: Denoise z_q (the full quantized content)
        # ----------------------------------------------------------------
        z_q_denoised, zc2_pred = self._run_denoiser(z_q, phone_ids, strength)

        # ----------------------------------------------------------------
        # Step 5: Recompute zc2
        # ----------------------------------------------------------------
        zc2 = self.zc2_recomputer.recompute(
            zc1_denoised=z_q_denoised,
            zc2_pred=zc2_pred,
            z_q=z_q,
            z_p=z_p,
            z_r=z_r,
        )

        # ----------------------------------------------------------------
        # Step 6: Decode
        # ----------------------------------------------------------------
        output_latents = latents
        output_latents.content = z_q_denoised
        output_latents.content_zc2 = zc2

        output_wav = self.facodec_adapter.decode(output_latents)

        return {
            "z_q": z_q.cpu(),
            "z_c1_original": z_c1_original.cpu(),
            "z_p": z_p.cpu(),
            "z_r": z_r.cpu(),
            "g": g.cpu() if g is not None else None,
            "phone_ids": phone_ids.cpu(),
            "z_q_denoised": z_q_denoised.cpu(),
            "zc2_pred": zc2_pred.cpu(),
            "output_wav": output_wav.cpu(),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_denoiser(
        self,
        z_q: torch.Tensor,
        phone_ids: torch.Tensor,
        strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the denoiser and return denoised z_q + predicted zc2.

        The denoiser's forward() returns (eps_pred, zc2_pred) where:
          - eps_pred: predicted noise epsilon [B, C, T]
          - zc2_pred: predicted content residual [B, C, T]

        DDPM inference:
          1. Map strength to timestep t_start
          2. Add noise: z_noisy = sqrt(abar[t]) * z_q + sqrt(1-abar[t]) * noise
          3. Denoise: eps_pred = denoiser(z_noisy, phone_ids, t)
          4. Recover x0: x0_hat = (z_noisy - sqrt(1mabar[t]) * eps_pred) / sqrt(abar[t])
          5. Interpolate: z_q_denoised = (1-strength) * z_q + strength * x0_hat
        """
        bsz, feat_dim, seq_len = z_q.shape

        # Map strength to diffusion timestep
        num_steps = getattr(self.denoiser, "num_steps", 100)
        t_start = int(round(strength * num_steps))
        t_start = max(0, min(num_steps - 1, t_start))

        if t_start == 0:
            # No denoising needed -- pass through
            # Still run denoiser at t=0 to get zc2 prediction
            t = torch.tensor([0], device=self.device).repeat(bsz)
            eps_pred, zc2_pred = self.denoiser(z_q, phone_ids, t)
            z_q_denoised = z_q
            return z_q_denoised, zc2_pred

        # Get DDPM schedule buffers from the denoiser
        sqrt_abar = self.denoiser.sqrt_abar
        sqrt_1mabar = self.denoiser.sqrt_1m_abar

        # Add noise: z_noisy = sqrt(abar[t]) * z_q + sqrt(1-abar[t]) * noise
        t_tensor = torch.tensor([t_start], device=self.device).repeat(bsz)
        noise = torch.randn_like(z_q)
        sa = sqrt_abar[t_start].view(1, 1, 1)
        s1a = sqrt_1mabar[t_start].view(1, 1, 1)
        z_noisy = sa * z_q + s1a * noise

        # Run denoiser
        eps_pred, zc2_pred = self.denoiser(z_noisy, phone_ids, t_tensor)

        # Recover x0 prediction
        x0_hat = (z_noisy - s1a * eps_pred) / sa

        # Interpolate between original and denoised based on strength
        z_q_denoised = (1.0 - strength) * z_q + strength * x0_hat

        return z_q_denoised, zc2_pred

    def _validate_latent_shapes(
        self,
        z_q: torch.Tensor,
        z_c1: torch.Tensor,
        z_p: torch.Tensor,
        z_r: torch.Tensor,
        g: torch.Tensor | None,
    ) -> None:
        """Validate that latent shapes are consistent."""
        bsz, feat_dim, seq_len = z_q.shape

        assert z_c1.shape[0] == bsz, (
            f"z_c1 batch size {z_c1.shape[0]} != z_q batch size {bsz}"
        )
        assert z_c1.shape[2] == seq_len, (
            f"z_c1 frame count {z_c1.shape[2]} != z_q frame count {seq_len}"
        )
        assert z_p.shape[0] == bsz, (
            f"z_p batch size {z_p.shape[0]} != z_q batch size {bsz}"
        )
        assert z_p.shape[2] == seq_len, (
            f"z_p frame count {z_p.shape[2]} != z_q frame count {seq_len}"
        )
        assert z_r.shape[0] == bsz, (
            f"z_r batch size {z_r.shape[0]} != z_q batch size {bsz}"
        )
        assert z_r.shape[2] == seq_len, (
            f"z_r frame count {z_r.shape[2]} != z_q frame count {seq_len}"
        )

        if g is not None:
            assert g.shape[0] == bsz, (
                f"g batch size {g.shape[0]} != z_q batch size {bsz}"
            )

    def _assert_frame_rate_contract(
        self,
        phone_ids: torch.Tensor,
        z_q: torch.Tensor,
    ) -> None:
        """Assert that phone_ids and z_q have matching frame counts."""
        phone_frames = phone_ids.shape[-1]
        z_frames = z_q.shape[-1]

        if phone_frames != z_frames:
            raise ValueError(
                f"Frame rate contract violated: phone_ids has {phone_frames} "
                f"frames but z_q has {z_frames} frames. "
                f"The PhonemePipeline must output 80fps phone IDs matching "
                f"the FACodec frame rate."
            )
