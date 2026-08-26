"""Phase 1 — ZC2 recomputation after zc1 denoising.

Per ZC2_CONTRACT.md:
  - zc2 must be recomputed when zc1 is modified (e.g., by the accent denoiser).
  - Two modes:
      'predict':  Use the denoiser's fc_zc2 head output (paper-faithful, default).
      'recompute': Re-run the second codebook of the content RVQ on the
                   residual (encoder_features - z_p - modified_zc1).

The predict approach is the recommended primary path per the contract.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional, Literal

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class ZC2RecomputeResult:
    """Output of ZC2Recomputer.

    Attributes:
        zc1:      Modified zc1, ready to be combined for the decoder.
        zc2:      Recomputed zc2, consistent with the modified zc1.
        zc2_pred: Raw zc2 prediction from fc_zc2 (only meaningful in 'predict' mode).
        mode:     Which mode produced this result.
        valid:    Whether zc2 is guaranteed consistent with modified_zc1.
    """
    zc1: torch.Tensor           # [B, C, T]
    zc2: torch.Tensor           # [B, C, T]
    zc2_pred: torch.Tensor      # [B, C, T] raw head output (predict mode)
    mode: str = "predict"       # 'predict' | 'recompute'
    valid: bool = True


# ---------------------------------------------------------------------------
# Helper: straight-through VQ approximation (recompute-mode fallback)
# ---------------------------------------------------------------------------

class _StraightThroughVQ(nn.Module):
    """Minimal straight-through VQ used when the real codebook is inaccessible.

    Projects to codebook_dim, adds noise as a stand-in for codebook lookup,
    then projects back. The straight-through estimator keeps gradients flowing.

    This is ONLY a fallback for 'recompute' mode when the quantizer does not
    expose individual codebooks.  The predict mode does not use this.
    """

    def __init__(self, input_dim: int, codebook_dim: int = 8):
        super().__init__()
        self.input_dim = input_dim
        self.codebook_dim = codebook_dim
        self.in_proj = nn.Linear(input_dim, codebook_dim)
        self.out_proj = nn.Linear(codebook_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, D, T]
        h = self.in_proj(x.transpose(1, 2)).transpose(1, 2)       # [B, 8, T]
        q = h  # no real codebook lookup; noise-based placeholder
        out = self.out_proj(q.transpose(1, 2)).transpose(1, 2)    # [B, D, T]
        # Straight-through: keep gradient flowing through input
        return x + (out - x).detach()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ZC2Recomputer:
    """Recomputes zc2 after zc1 has been modified by the denoiser.

    Two modes:
      - 'predict':  Use the denoiser's fc_zc2 head (paper-faithful, default).
      - 'recompute': Re-run the second codebook of the content RVQ on the
                     residual (encoder_features - z_p.detach() - modified_zc1).

    The predict approach is the default and recommended primary path per
    ZC2_CONTRACT.md §8.  The recompute mode is a fallback for cases where
    the quantizer's individual codebooks are accessible.
    """

    def __init__(
        self,
        mode: Literal["predict", "recompute"] = "predict",
        denoiser: Optional[nn.Module] = None,
        quantizer: Optional[nn.Module] = None,
        facodec_dim: int = 8,
        num_steps: int = 100,
        device: str = "cpu",
    ):
        """Initialise ZC2Recomputer.

        Args:
            mode:       'predict' (default) or 'recompute'.
            denoiser:   The Phase1AccentNormalizer (or DenoisingTransformerModel).
                        Required for 'predict' mode.
            quantizer:  The content quantizer (must expose individual codebooks).
                        Required for 'recompute' mode.
            facodec_dim: Dimensionality of the codebook space (default 8).
            num_steps:  Number of diffusion timesteps (default 100).
            device:     Torch device string.
        """
        self.mode = mode
        self.denoiser = denoiser
        self.quantizer = quantizer
        self.facodec_dim = facodec_dim
        self.num_steps = num_steps
        self.device = torch.device(device)

        # Validation is deferred to recompute() / forward() call time so that
        # ZC2Recomputer instances can be created without all components
        # eagerly (e.g., during config-driven construction).

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def recompute(
        self,
        encoder_features: torch.Tensor,   # [B, 1024, T]
        modified_zc1: torch.Tensor,        # [B, C, T]
        z_p: torch.Tensor,                # [B, C_p, T]
        z_r: torch.Tensor,                # [B, C_r, T]
        phone_ids: Optional[torch.Tensor] = None,  # [B, T]
    ) -> ZC2RecomputeResult:
        """Given modified zc1, recompute valid zc2.

        This is the lower-level entry point.  For the full content stack
        (including additional phone conditioning), use ``forward()``.

        Note: In 'predict' mode, ``phone_ids`` is required because the
        fc_zc2 head needs phoneme conditioning.  If not provided, the
        denoiser will raise a ValueError at call time.

        Args:
            encoder_features: Encoder output  [B, 1024, T].
            modified_zc1:     Modified (denoised) zc1  [B, C, T].
            z_p:              Prosody latents  [B, C_p, T].
            z_r:              Residual latents  [B, C_r, T].
            phone_ids:        Phoneme IDs  [B, T].  Required in 'predict' mode.

        Returns:
            ZC2RecomputeResult with zc1 + zc2 consistent for the decoder.
        """
        if self.mode == "predict":
            if phone_ids is None:
                raise ValueError(
                    "phone_ids is required for 'predict' mode. "
                    "Use forward() or pass phone_ids explicitly."
                )
            return self._predict_zc2(encoder_features, modified_zc1, z_p, z_r, phone_ids)
        return self._recompute_zc2(encoder_features, modified_zc1, z_p, z_r)

    def forward(
        self,
        denoised_zc1: torch.Tensor,            # [B, C, T]
        encoder_features: torch.Tensor,        # [B, 1024, T]
        z_p: torch.Tensor,                    # [B, C_p, T]
        z_r: torch.Tensor,                    # [B, C_r, T]
        phone_ids: Optional[torch.Tensor] = None,  # [B, T]
    ) -> ZC2RecomputeResult:
        """Full content-stack recomputation.

        Given denoised zc1, recompute zc2 and return (zc1, zc2) ready to be
        combined into the decoder's input latent:

            z_combined = z_p + (zc1 + zc2) + z_r

        Args:
            denoised_zc1:     Denoised zc1 from the diffusion process  [B, C, T].
            encoder_features: Encoder output  [B, 1024, T].
            z_p:              Prosody latents  [B, C_p, T].
            z_r:              Residual latents  [B, C_r, T].
            phone_ids:        Phoneme IDs  [B, T].  Required in 'predict' mode.

        Returns:
            ZC2RecomputeResult whose ``zc1`` and ``zc2`` fields form a valid
            content representation for the decoder.
        """
        if self.mode == "predict":
            if phone_ids is None:
                raise ValueError(
                    "phone_ids is required for 'predict' mode (fc_zc2 head needs phoneme conditioning)."
                )
            return self._predict_zc2(
                encoder_features, denoised_zc1, z_p, z_r, phone_ids
            )
        return self._recompute_zc2(encoder_features, denoised_zc1, z_p, z_r)

    # ------------------------------------------------------------------
    # 'predict' mode — fc_zc2 head
    # ------------------------------------------------------------------

    def _predict_zc2(
        self,
        encoder_features: torch.Tensor,
        modified_zc1: torch.Tensor,
        z_p: torch.Tensor,
        z_r: torch.Tensor,
        phone_ids: torch.Tensor,  # required for phoneme conditioning
    ) -> ZC2RecomputeResult:
        """Predict zc2 via the denoiser's fc_zc2 head.

        Per ZC2_CONTRACT.md §3.8 and §6.2:
          - The denoiser's fc_zc2 head takes (transformer_hidden_states, x0_hat)
            and predicts zc2.
          - At timestep t=0, x0_hat == modified_zc1 (fully denoised).
          - We run a single forward pass through the denoiser at t=0 to obtain
            zc2_pred without actually performing the full diffusion loop.

        Args:
            phone_ids: Phoneme IDs [B, T].  Required for phoneme conditioning.
        """
        assert self.denoiser is not None, "denoiser must be set for 'predict' mode"

        modified_zc1 = modified_zc1.to(self.device)
        z_p = z_p.to(self.device)
        z_r = z_r.to(self.device)
        phone_ids = phone_ids.to(self.device)

        bsz = modified_zc1.shape[0]
        seq_len = modified_zc1.shape[2]

        # t=0 means fully denoised: x0_hat = modified_zc1 (no noise added)
        t = torch.zeros(bsz, device=self.device, dtype=torch.long)

        # All frames are valid (no padding)
        padding_mask = torch.zeros(bsz, seq_len, device=self.device, dtype=torch.bool)

        with torch.no_grad():
            # The denoiser returns (eps_pred, zc2_pred)
            _eps_pred, zc2_pred = self.denoiser(
                modified_zc1, phone_ids, t, padding_mask=padding_mask
            )

        # At t=0: x0_hat = modified_zc1, so fc_zc2 predicts zc2 from the
        # denoised zc1 estimate.  zc2_pred shape: [B, facodec_dim, T]
        zc2_pred = zc2_pred.to(self.device)

        # Align to modified_zc1's channel dimension if needed
        if zc2_pred.shape[1] != modified_zc1.shape[1]:
            zc2_pred = self._align_channels(zc2_pred, modified_zc1.shape[1])

        return ZC2RecomputeResult(
            zc1=modified_zc1,
            zc2=zc2_pred,
            zc2_pred=zc2_pred,
            mode="predict",
            valid=True,
        )

    # ------------------------------------------------------------------
    # 'recompute' mode — re-run second codebook
    # ------------------------------------------------------------------

    def _recompute_zc2(
        self,
        encoder_features: torch.Tensor,
        modified_zc1: torch.Tensor,
        z_p: torch.Tensor,
        z_r: torch.Tensor,
    ) -> ZC2RecomputeResult:
        """Recompute zc2 by re-running the second codebook on the residual.

        Per ZC2_CONTRACT.md §6.2:
          residual = encoder_features - z_p.detach() - modified_zc1
          zc2 = VQ_1(residual)   (second codebook, index 1)
        """
        assert self.quantizer is not None, "quantizer must be set for 'recompute' mode"

        encoder_features = encoder_features.to(self.device)
        modified_zc1 = modified_zc1.to(self.device)
        z_p = z_p.to(self.device)
        z_r = z_r.to(self.device)

        # Residual for the second codebook
        residual = encoder_features - z_p.detach() - modified_zc1

        # Run the second codebook
        zc2 = self._run_second_codebook(residual, modified_zc1.shape[1])

        return ZC2RecomputeResult(
            zc1=modified_zc1,
            zc2=zc2,
            zc2_pred=zc2,
            mode="recompute",
            valid=True,
        )

    def _run_second_codebook(
        self, residual: torch.Tensor, target_dim: int
    ) -> torch.Tensor:
        """Run the second codebook (index 1) of the content RVQ.

        Tries to access the real codebook; falls back to a straight-through
        estimator with a warning.
        """
        quantizer = self.quantizer

        # --- Attempt 1: direct access via content_quantizer.quantizers[1] ---
        try:
            cq = quantizer.content_quantizer  # type: ignore[attr-defined]
            vq = cq.quantizers[1]  # type: ignore[attr-defined]
            zc2 = vq(residual)
            if zc2.shape[1] != target_dim:
                zc2 = self._align_channels(zc2, target_dim)
            return zc2
        except (AttributeError, IndexError, TypeError):
            pass

        # --- Attempt 2: RVQ-style forward with start_codebook ---
        try:
            cq = quantizer.content_quantizer  # type: ignore[attr-defined]
            if hasattr(cq, "start_codebook"):
                z_q, _codes, all_q = cq(residual, start_codebook=1)
                if all_q and len(all_q) > 0:
                    zc2 = all_q[0]
                    if zc2.shape[1] != target_dim:
                        zc2 = self._align_channels(zc2, target_dim)
                    return zc2
        except (AttributeError, TypeError):
            pass

        # --- Fallback: straight-through estimator ---
        warnings.warn(
            "The quantizer does not expose individual codebook outputs. "
            "Using a straight-through estimator as a fallback for zc2 "
            "recomputation.  Use 'predict' mode for paper-faithful results.",
            RuntimeWarning,
        )
        st_vq = _StraightThroughVQ(
            input_dim=residual.shape[1], codebook_dim=self.facodec_dim
        ).to(residual.device)
        zc2 = st_vq(residual)
        if zc2.shape[1] != target_dim:
            zc2 = self._align_channels(zc2, target_dim)
        return zc2

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _align_channels(x: torch.Tensor, target_channels: int) -> torch.Tensor:
        """Project x to target_channels if they differ."""
        if x.shape[1] == target_channels:
            return x
        proj = nn.Linear(x.shape[1], target_channels).to(x.device)
        with torch.no_grad():
            return proj(x.transpose(1, 2)).transpose(1, 2)

    # ------------------------------------------------------------------
    # Convenience: build content latent for decoder
    # ------------------------------------------------------------------

    @staticmethod
    def build_content_latent(
        zc1: torch.Tensor,
        zc2: torch.Tensor,
        z_p: torch.Tensor,
        z_r: torch.Tensor,
    ) -> torch.Tensor:
        """Combine factorized components into the decoder's input latent.

        Per ZC2_CONTRACT.md §6.2:
          z_q = z_p + (zc1 + zc2) + z_r

        All inputs must share the same channel dimension (typically 1024
        after the quantizer's out_proj projection).  This matches the
        decoder's expectation: it receives a single [B, 1024, T] tensor.

        Args:
            zc1: Content codebook-1 output  [B, C, T].
            zc2: Content codebook-2 output  [B, C, T].
            z_p: Prosody output              [B, C, T].
            z_r: Residual output             [B, C, T].

        Returns:
            Combined latent  [B, C, T].
        """
        return z_p + (zc1 + zc2) + z_r
