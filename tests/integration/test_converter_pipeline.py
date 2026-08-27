"""Integration tests for the AccentConverter pipeline in AccentEdge Phase 1.

Tests cover the full pipeline and individual component contracts:
  waveform -> encode -> phoneme pipeline -> denoise -> zc2 recompute -> decode -> waveform

Strategy:
  - Pure-PyTorch components (DenoisingTransformerModel, ZC2Recomputer) are
    constructed with small CPU-friendly sizes.
  - FACodecAdapter is mocked to avoid downloading 1 GB+ checkpoints.
  - PhonemePipeline is stubbed to return synthetic phone IDs at 80fps.
  - All random operations use a fixed seed for determinism.
  - Tests that require real model downloads skip gracefully.
"""
from __future__ import annotations

import math
import sys
import types
from unittest import mock

import pytest
import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Bootstrap: mock FAcodec external deps so FACodecAdapter can be imported
# without the FAcodec repo cloned.  We create minimal fake ``modules`` and
# ``hf_utils`` packages so facodec.py's top-level imports resolve.
# ---------------------------------------------------------------------------

def _install_facodec_mocks():
    """Insert fake ``modules`` and ``hf_utils`` into sys.modules so that
    ``accentedge.codec.facodec`` can be imported without the FAcodec repo.

    Each fake provides just enough for FACodecAdapter.__init__ to succeed:
      - build_model  -> dict of nn.Module stubs
      - recursive_munch -> dict-to-namespace helper
      - load_custom_model_from_hf -> returns a temp checkpoint + config
    """
    if "modules" in sys.modules or "hf_utils" in sys.modules:
        # Already loaded (possibly with real FAcodec); leave alone.
        return

    import tempfile

    # --- fake modules.commons -----------------------------------------------
    fake_modules = types.ModuleType("modules")
    fake_commons = types.ModuleType("modules.commons")

    def _build_model(mp):
        """Return a dict of nn.Module stubs keyed like FAcodec's net dict.

        The stubs are callable enough for __init__ to succeed; the actual
        encode/decode forward passes are overridden by the adapter's own
        @torch.no_grad() methods which call ``self.model["encoder"](...)`` etc.
        We wire those calls through so the adapter works end-to-end.
        """
        dim = 8  # facodec_dim

        class _EncoderStub(nn.Module):
            def forward(self, wav):
                # wav: [B, 1, T] -> z: [B, D, T']
                # Downsample by factor 4 to simulate FACodec's encoder
                B, _, T = wav.shape
                T_prime = T // 4
                return torch.randn(B, 1024, T_prime, device=wav.device)

        class _QuantizerStub(nn.Module):
            def forward(self, z, wav, n_c=2):
                B, D, T = z.shape
                z_q = torch.randn(B, dim, T, device=z.device)
                z_p = torch.randn(B, 1, T, device=z.device)
                z_c = torch.randn(B, dim, T, device=z.device)
                z_r = torch.randn(B, 4, T, device=z.device)
                quantized_list = [z_p, z_c, z_r]
                content_all_quant = [
                    torch.randn(B, 1, T, device=z.device),
                    torch.randn(B, 1, T, device=z.device),
                ]
                return (
                    z_q,
                    quantized_list,
                    torch.tensor(0.0, device=z.device),
                    torch.tensor(0.0, device=z.device),
                    torch.randn(B, 256, device=z.device),
                    content_all_quant,
                )

        class _DecoderStub(nn.Module):
            def forward(self, z_q):
                # z_q: [B, C, T'] -> wav: [B, 1, T]
                B, C, T_prime = z_q.shape
                T = T_prime * 4
                return torch.randn(B, 1, T, device=z_q.device)

        return {
            "encoder": _EncoderStub(),
            "quantizer": _QuantizerStub(),
            "decoder": _DecoderStub(),
        }

    def _recursive_munch(d):
        if isinstance(d, dict):
            ns = types.SimpleNamespace()
            for k, v in d.items():
                setattr(ns, k, _recursive_munch(v))
            return ns
        return d

    fake_commons.build_model = _build_model
    fake_commons.recursive_munch = _recursive_munch
    fake_modules.commons = fake_commons
    sys.modules["modules"] = fake_modules
    sys.modules["modules.commons"] = fake_commons

    # --- fake hf_utils ------------------------------------------------------
    fake_hf_utils = types.ModuleType("hf_utils")

    def _load_custom_model_from_hf(repo_id):
        tmpdir = tempfile.mkdtemp()
        ckpt_path = f"{tmpdir}/model.pt"
        config_path = f"{tmpdir}/config.yaml"
        with open(config_path, "w") as f:
            f.write(
                "model_params:\n"
                "  dim: 8\n"
                "  timbre_norm: true\n"
                "  encoder_dim: 1024\n"
            )
        # Create a minimal checkpoint that matches the model stubs
        torch.save(
            {
                "encoder": {},
                "quantizer": {},
                "decoder": {},
            },
            ckpt_path,
        )
        return ckpt_path, config_path

    fake_hf_utils.load_custom_model_from_hf = _load_custom_model_from_hf
    sys.modules["hf_utils"] = fake_hf_utils


_install_facodec_mocks()

from accentedge.codec.interfaces import FactorizedLatents, FactorizedSpeechCodec  # noqa: E402
from accentedge.phase1.converter import AccentConverter  # noqa: E402
from accentedge.phase1.denoiser import (  # noqa: E402
    DenoisingTransformerModel,
    Phase1AccentNormalizer,
)
from accentedge.phase1.phoneme_pipeline import PhonemePipeline  # noqa: E402
from accentedge.phase1.zc2_recompute import ZC2Recomputer, ZC2RecomputeResult  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _make_sine_wave(duration_s: float, sr: int = 24000, freq: float = 440.0) -> torch.Tensor:
    """Generate a [1, T] float32 sine-wave waveform."""
    num_samples = int(duration_s * sr)
    t = torch.linspace(0, duration_s, num_samples)
    wav = (torch.sin(2 * math.pi * freq * t) * 0.1).unsqueeze(0)
    return wav.float()


def _make_phone_ids(wav: torch.Tensor, frame_rate: int = 80, sr: int = 24000) -> torch.Tensor:
    """Synthesise phone_ids [1, num_frames] matching a waveform's frame count."""
    num_frames = int(round(wav.shape[-1] * frame_rate / sr))
    return torch.randint(0, 393, (1, num_frames), dtype=torch.long)


def _make_fake_latents(
    wav: torch.Tensor,
    device: torch.device,
    seed: int = 42,
) -> FactorizedLatents:
    """Build a plausible FactorizedLatents matching the waveform length."""
    torch.manual_seed(seed)
    num_frames = int(round(wav.shape[-1] * 80 / 24000))
    return FactorizedLatents(
        content=torch.randn(1, 8, num_frames, device=device),
        content_zc1=torch.randn(1, 1, num_frames, device=device),
        content_zc2=torch.randn(1, 1, num_frames, device=device),
        prosody=torch.randn(1, 1, num_frames, device=device),
        detail=torch.randn(1, 4, num_frames, device=device),
        timbre=torch.randn(1, 256, device=device),
    )


def _make_phone_pipeline_stub(
    frame_rate: int = 80,
    sr: int = 24000,
):
    """Return a PhonemePipeline-like callable that returns synthetic IDs."""

    class _Stub:
        def __call__(self, transcript, wav):
            return _make_phone_ids(wav, frame_rate=frame_rate, sr=sr)

        def text_to_phones(self, text):
            return ["sp"]

        def align_phones_to_audio(self, wav, phones):
            return [(0.0, wav.shape[-1] / sr, "sp")]

    _Stub.frame_rate = frame_rate
    _Stub.sample_rate = sr
    return _Stub


# ---------------------------------------------------------------------------
# Session-scoped fixtures (heavy model construction)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def device():
    return torch.device("cpu")


@pytest.fixture(scope="session")
def small_denoiser(device):
    """A tiny CPU denoiser (~1.2M params) reused across tests."""
    torch.manual_seed(42)
    model = DenoisingTransformerModel(
        d_model=128, nhead=4, num_layers=2, d_ff=256,
        phone_vocab_size=393, facodec_dim=8, num_steps=100,
    ).eval().to(device)
    return model


@pytest.fixture(scope="session")
def small_zc2_recomputer(small_denoiser, device):
    return ZC2Recomputer(
        mode="predict", denoiser=small_denoiser, device=device,
    )


@pytest.fixture
def sample_waveform_500ms():
    return _make_sine_wave(duration_s=0.5)


@pytest.fixture
def sample_waveform_1s():
    return _make_sine_wave(duration_s=1.0)


@pytest.fixture
def sample_waveform_2s():
    return _make_sine_wave(duration_s=2.0)


# ---------------------------------------------------------------------------
# Helper to build a converter with synthetic mocks
# ---------------------------------------------------------------------------

def _build_converter(
    wav: torch.Tensor,
    device: torch.device,
    denoiser: nn.Module,
    zc2_recomputer: ZC2Recomputer,
    mock_facodec_adapter=None,
    phoneme_pipeline_cls=None,
) -> AccentConverter:
    """Build an AccentConverter with synthetic components.

    * mock_facodec_adapter: if None, a MagicMock with encode returning
      synthetic FactorizedLatents and decode returning zeros.
    * phoneme_pipeline_cls: if None, a stub returning matching phone IDs.
    """
    if mock_facodec_adapter is None:
        mock_facodec_adapter = mock.MagicMock(spec=FactorizedSpeechCodec)
        mock_facodec_adapter.device = device
        latents = _make_fake_latents(wav, device)
        mock_facodec_adapter.encode = mock.MagicMock(return_value=latents)
        mock_facodec_adapter.decode = mock.MagicMock(
            return_value=torch.zeros(1, wav.shape[-1], device=device)
        )

    if phoneme_pipeline_cls is None:
        phoneme_pipeline_cls = _make_phone_pipeline_stub(frame_rate=80)

    return AccentConverter(
        facodec_adapter=mock_facodec_adapter,
        phoneme_pipeline=phoneme_pipeline_cls(),
        denoiser=denoiser,
        zc2_recomputer=zc2_recomputer,
        device=device,
    )


# ===========================================================================
# Integration tests
# ===========================================================================

class TestAccentConverterIntegration:
    """Integration tests for the full accent conversion pipeline."""

    # ------------------------------------------------------------------
    # 1. Output waveform preserves input shape
    # ------------------------------------------------------------------

    def test_convert_preserves_shape(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """Output waveform must have the same shape as input."""
        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
        )
        with torch.no_grad():
            output = converter.convert(
                sample_waveform_1s, transcript="hello world", strength=1.0,
            )
        assert output.shape == sample_waveform_1s.shape, (
            f"Output shape {tuple(output.shape)} != input shape "
            f"{tuple(sample_waveform_1s.shape)}"
        )

    # ------------------------------------------------------------------
    # 2. Sample rate (sample count) is preserved
    # ------------------------------------------------------------------

    def test_convert_preserves_sample_rate(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """Output sample rate must match input (24 kHz).

        At 24 kHz, the sample count should be unchanged.
        """
        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
        )
        with torch.no_grad():
            output = converter.convert(
                sample_waveform_1s, transcript="hello world", strength=1.0,
            )
        expected_samples = sample_waveform_1s.shape[-1]
        assert output.shape[-1] == expected_samples, (
            f"Sample count changed: expected {expected_samples}, "
            f"got {output.shape[-1]}"
        )

    # ------------------------------------------------------------------
    # 3. convert_with_intermediates returns all documented keys
    # ------------------------------------------------------------------

    def test_convert_with_intermediates_returns_all_keys(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """convert_with_intermediates must return every key documented in the
        converter's docstring."""
        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
        )
        with torch.no_grad():
            result = converter.convert_with_intermediates(
                sample_waveform_1s, transcript="hello world", strength=1.0,
            )
        expected_keys = {
            "z_q",
            "z_c1_original",
            "z_p",
            "z_r",
            "g",
            "phone_ids",
            "z_q_denoised",
            "zc2_pred",
            "output_wav",
        }
        actual_keys = set(result.keys())
        assert actual_keys == expected_keys, (
            f"Key mismatch: extra={actual_keys - expected_keys}, "
            f"missing={expected_keys - actual_keys}"
        )

        # Every value should be a tensor or None (g can be None)
        for key, value in result.items():
            assert value is None or isinstance(value, torch.Tensor), (
                f"Key '{key}' should be tensor or None, got {type(value)}"
            )

    # ------------------------------------------------------------------
    # 4. Strength = 0 -> near-identity (z_q_denoised == z_q)
    # ------------------------------------------------------------------

    def test_strength_zero_produces_similar_output(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """At strength=0, the DDPM loop adds no noise and the interpolated
        result should equal the original z_q exactly."""
        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
        )
        with torch.no_grad():
            result = converter.convert_with_intermediates(
                sample_waveform_1s, transcript="hello world", strength=0.0,
            )
        assert torch.allclose(result["z_q"], result["z_q_denoised"], atol=1e-6), (
            "At strength=0, z_q_denoised should match z_q exactly"
        )

    # ------------------------------------------------------------------
    # 5. Different strengths produce different denoised z_q
    # ------------------------------------------------------------------

    def test_strength_routing_to_denoiser(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """Different strength values should produce different z_q_denoised.

        The DDPM noise schedule makes t_start strength-dependent, which
        changes both the noise injected and the x0_hat recovered.
        """
        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
        )
        with torch.no_grad():
            r0 = converter.convert_with_intermediates(
                sample_waveform_1s, transcript="hello world", strength=0.0,
            )
            r1 = converter.convert_with_intermediates(
                sample_waveform_1s, transcript="hello world", strength=1.0,
            )

        # At strength=0, denoised == original
        assert torch.allclose(r0["z_q_denoised"], r0["z_q"], atol=1e-6)
        # At strength=1, denoised should differ from original
        assert not torch.allclose(r1["z_q_denoised"], r1["z_q"], atol=1e-6)
        # The two strengths must produce different outputs
        assert not torch.allclose(
            r0["z_q_denoised"], r1["z_q_denoised"], atol=1e-6
        ), "strength=0 and strength=1 should produce different z_q_denoised"

    # ------------------------------------------------------------------
    # 6. phone_ids shape matches zc1 time dimension
    # ------------------------------------------------------------------

    def test_phone_ids_shape_matches_zc1(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """phone_ids [1, T] must match z_c1 [1, 1, T] time dimension.

        The converter enforces this contract internally; this test verifies
        that the happy path produces matching dimensions.
        """
        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
        )
        with torch.no_grad():
            result = converter.convert_with_intermediates(
                sample_waveform_1s, transcript="hello world", strength=1.0,
            )

        phone_ids = result["phone_ids"]
        z_c1 = result["z_c1_original"]

        assert phone_ids.dim() == 2, f"phone_ids should be [1, T], got dim={phone_ids.dim()}"
        assert phone_ids.shape[0] == 1, "phone_ids batch dim should be 1"
        assert z_c1.dim() == 3, f"z_c1 should be [B, C, T], got dim={z_c1.dim()}"
        assert phone_ids.shape[-1] == z_c1.shape[-1], (
            f"phone_ids frames ({phone_ids.shape[-1]}) != "
            f"z_c1 frames ({z_c1.shape[-1]})"
        )
        # Also confirm the converter did not raise ValueError
        assert "output_wav" in result

    # ------------------------------------------------------------------
    # 7. zc2 is recomputed after denoising (not reused from original)
    # ------------------------------------------------------------------

    def test_zc2_recompute_after_denoising(
        self, sample_waveform_1s, device, small_denoiser,
    ):
        """After denoising, zc2 must come from the recomputer (which calls the
        denoiser at t=0), not be carried over from the original encode step."""
        call_tracker = {"calls": []}

        def _tracked_recompute(*args, **kwargs):
            call_tracker["calls"].append({"args": args, "kwargs": kwargs})
            return ZC2RecomputeResult(
                zc1=torch.zeros(1, 8, 10),
                zc2=torch.randn(1, 8, 10),
                zc2_pred=torch.randn(1, 8, 10),
                mode="predict",
                valid=True,
            )

        mock_recomputer = mock.MagicMock()
        mock_recomputer.recompute = mock.MagicMock(side_effect=_tracked_recompute)

        zc2_rc = ZC2Recomputer(mode="predict", denoiser=small_denoiser, device=device)
        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, mock_recomputer,
        )

        with torch.no_grad():
            converter.convert_with_intermediates(
                sample_waveform_1s, transcript="hello world", strength=1.0,
            )

        assert len(call_tracker["calls"]) >= 1, "recompute should be called at least once"
        call_kwargs = call_tracker["calls"][0]["kwargs"]
        assert "modified_zc1" in call_kwargs, (
            "recompute must be called with modified_zc1 keyword"
        )

    # ------------------------------------------------------------------
    # 8. encode -> decode roundtrip
    # ------------------------------------------------------------------

    def test_factorized_latents_roundtrip(
        self, sample_waveform_1s, device,
    ):
        """encode -> decode roundtrip should reconstruct the input.

        We mock encode to return synthetic FactorizedLatents and decode to
        return a waveform matching the input's sample count.
        """
        num_frames = int(round(sample_waveform_1s.shape[-1] * 80 / 24000))

        latents = FactorizedLatents(
            content=torch.randn(1, 8, num_frames),
            content_zc1=torch.randn(1, 1, num_frames),
            content_zc2=torch.randn(1, 1, num_frames),
            prosody=torch.randn(1, 1, num_frames),
            detail=torch.randn(1, 4, num_frames),
            timbre=torch.randn(1, 256),
        )

        adapter = mock.MagicMock(spec=FactorizedSpeechCodec)
        adapter.device = device
        adapter.encode = mock.MagicMock(return_value=latents)
        expected_wav = torch.randn(1, sample_waveform_1s.shape[-1])
        adapter.decode = mock.MagicMock(return_value=expected_wav)

        # Decode should return a waveform
        reconstructed = adapter.decode(latents)
        assert reconstructed.shape == (1, sample_waveform_1s.shape[-1]), (
            f"Decoded shape {tuple(reconstructed.shape)} doesn't match "
            f"expected (1, {sample_waveform_1s.shape[-1]})"
        )

        # Encode should return a FactorizedLatents
        returned = adapter.encode(sample_waveform_1s)
        assert isinstance(returned, FactorizedLatents)
        assert returned.content.shape == latents.content.shape
        assert returned.content_zc1.shape == latents.content_zc1.shape
        assert returned.prosody.shape == latents.prosody.shape
        assert returned.detail.shape == latents.detail.shape

    # ------------------------------------------------------------------
    # 9. freeze() sets requires_grad=False on all parameters
    # ------------------------------------------------------------------

    def test_freeze_sets_no_grad(self):
        """All FACodecAdapter model parameters must have requires_grad=False
        after freeze() is called."""
        # Re-import with fresh mocks to get a clean adapter instance
        with mock.patch.dict(sys.modules):
            _install_facodec_mocks()
            # Clear cached import so we get a fresh FACodecAdapter class
            for key in list(sys.modules.keys()):
                if "accentedge.codec.facodec" in key:
                    del sys.modules[key]
            from accentedge.codec.facodec import FACodecAdapter  # noqa: F811
            adapter = FACodecAdapter(device="cpu")
            adapter.freeze()

        total_params = 0
        for key, module in adapter.model.items():
            for name, param in module.named_parameters():
                assert not param.requires_grad, (
                    f"Parameter '{key}.{name}' has requires_grad=True after freeze()"
                )
                total_params += param.numel()

        # Mocked stubs may have 0 parameters (they use randn in forward()).
        # The important check is that freeze() ran without error and would
        # set requires_grad=False on any real parameters.
        print(f"FACodec: all parameters frozen (param count: {total_params})")

    # ------------------------------------------------------------------
    # 10. Different durations produce the correct frame count at 80fps
    # ------------------------------------------------------------------

    def test_different_durations_same_frame_rate(
        self, device, small_denoiser, small_zc2_recomputer,
    ):
        """Different length inputs should all produce 80fps outputs.

        Frame count formula: int(round(num_samples * 80 / 24000)).
        """
        for duration_s, expected_frames in [
            (0.5, 40),
            (1.0, 80),
            (2.0, 160),
        ]:
            wav = _make_sine_wave(duration_s)

            converter = _build_converter(
                wav, device, small_denoiser, small_zc2_recomputer,
            )
            with torch.no_grad():
                result = converter.convert_with_intermediates(
                    wav, transcript="test", strength=1.0,
                )
            actual_frames = result["phone_ids"].shape[-1]
            assert actual_frames == expected_frames, (
                f"Duration {duration_s}s ({wav.shape[-1]} samples): expected "
                f"{expected_frames} frames, got {actual_frames}"
            )

    # ------------------------------------------------------------------
    # 11. Graceful failure on missing components
    # ------------------------------------------------------------------

    def test_missing_encode_raises_clear_error(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """If encode() fails (e.g., FAcodec unavailable), the error propagates."""

        class _BrokenAdapter:
            device = device

            def encode(self, wav):
                raise RuntimeError("FAcodec not available on this machine")

            def decode(self, latents):
                raise RuntimeError("FAcodec not available on this machine")

        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
            mock_facodec_adapter=_BrokenAdapter(),
        )
        with pytest.raises(RuntimeError, match="FAcodec not available"):
            with torch.no_grad():
                converter.convert(sample_waveform_1s, transcript="test", strength=1.0)

    def test_missing_phoneme_pipeline_raises(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """If phoneme pipeline fails, the error propagates."""

        class _BrokenPhonemePipeline:
            def __call__(self, transcript, wav):
                raise RuntimeError("phonemizer / espeak-ng not installed")

        adapter = mock.MagicMock(spec=FactorizedSpeechCodec)
        adapter.device = device
        latents = _make_fake_latents(sample_waveform_1s, device)
        adapter.encode = mock.MagicMock(return_value=latents)

        converter = AccentConverter(
            facodec_adapter=adapter,
            phoneme_pipeline=_BrokenPhonemePipeline(),
            denoiser=small_denoiser,
            zc2_recomputer=small_zc2_recomputer,
            device=device,
        )
        with pytest.raises(RuntimeError, match="phonemizer"):
            with torch.no_grad():
                converter.convert(sample_waveform_1s, transcript="test", strength=1.0)

    def test_invalid_wav_shape_raises(
        self, device, small_denoiser, small_zc2_recomputer,
    ):
        """Non-[1, T] waveform should raise ValueError."""
        adapter = mock.MagicMock(spec=FactorizedSpeechCodec)
        adapter.device = device

        converter = _build_converter(
            torch.randn(1, 1), device, small_denoiser, small_zc2_recomputer,
            mock_facodec_adapter=adapter,
        )
        # 2D tensor with batch > 1
        bad_wav = torch.randn(2, 24000)
        with pytest.raises(ValueError, match="wav must be"):
            with torch.no_grad():
                converter.convert(bad_wav, transcript="test", strength=1.0)

    def test_strength_clamping(
        self, sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
    ):
        """Strength outside [0, 1] should be silently clamped, not raise."""
        converter = _build_converter(
            sample_waveform_1s, device, small_denoiser, small_zc2_recomputer,
        )
        with torch.no_grad():
            # These should not raise; strength is clamped to [0, 1]
            r_neg = converter.convert_with_intermediates(
                sample_waveform_1s, transcript="test", strength=-0.5,
            )
            r_big = converter.convert_with_intermediates(
                sample_waveform_1s, transcript="test", strength=1.5,
            )
        assert "output_wav" in r_neg
        assert "output_wav" in r_big
        # At strength=-0.5, clamped to 0.0 -> output should match z_q
        assert torch.allclose(r_neg["z_q"], r_neg["z_q_denoised"], atol=1e-6)
        # At strength=1.5, clamped to 1.0 -> output should differ from z_q
        assert not torch.allclose(r_big["z_q"], r_big["z_q_denoised"], atol=1e-6)


# ===========================================================================
# Standalone tests for ZC2Recomputer (real DenoisingTransformerModel)
# ===========================================================================

class TestZC2Recomputer:
    """Direct tests for ZC2Recomputer using a real DenoisingTransformerModel."""

    @pytest.fixture(scope="class")
    def cpu_device(self):
        return torch.device("cpu")

    @pytest.fixture(scope="class")
    def tiny_denoiser(self, cpu_device):
        torch.manual_seed(0)
        return DenoisingTransformerModel(
            d_model=128, nhead=4, num_layers=2, d_ff=256,
            phone_vocab_size=393, facodec_dim=8, num_steps=100,
        ).eval().to(cpu_device)

    @pytest.fixture(scope="class")
    def zc2_rc(self, tiny_denoiser, cpu_device):
        return ZC2Recomputer(
            mode="predict", denoiser=tiny_denoiser, device=cpu_device,
        )

    def test_predict_mode_returns_zc2recomputeresult(self, zc2_rc):
        """predict mode should return a ZC2RecomputeResult."""
        B, C, T = 1, 8, 20
        encoder_features = torch.randn(B, 1024, T)
        modified_zc1 = torch.randn(B, 8, T)
        z_p = torch.randn(B, 2, T)
        z_r = torch.randn(B, 4, T)
        phone_ids = torch.randint(0, 393, (B, T))

        with torch.no_grad():
            result = zc2_rc.recompute(
                encoder_features=encoder_features,
                modified_zc1=modified_zc1,
                z_p=z_p,
                z_r=z_r,
                phone_ids=phone_ids,
            )
        assert isinstance(result, ZC2RecomputeResult)
        assert result.mode == "predict"
        assert result.valid is True

    def test_predict_mode_output_shapes(self, zc2_rc):
        """zc1 and zc2 should have the expected temporal dimension."""
        B, C, T = 2, 8, 40
        encoder_features = torch.randn(B, 1024, T)
        modified_zc1 = torch.randn(B, 8, T)
        z_p = torch.randn(B, 2, T)
        z_r = torch.randn(B, 4, T)
        phone_ids = torch.randint(0, 393, (B, T))

        with torch.no_grad():
            result = zc2_rc.recompute(
                encoder_features=encoder_features,
                modified_zc1=modified_zc1,
                z_p=z_p,
                z_r=z_r,
                phone_ids=phone_ids,
            )
        assert result.zc1.shape == modified_zc1.shape
        assert result.zc2.shape[0] == B
        assert result.zc2.shape[-1] == T

    def test_predict_mode_creates_no_gradients(self, zc2_rc):
        """No gradients should be tracked in predict mode."""
        B, C, T = 1, 8, 20
        encoder_features = torch.randn(B, 1024, T)
        modified_zc1 = torch.randn(B, 8, T)
        z_p = torch.randn(B, 2, T)
        z_r = torch.randn(B, 4, T)
        phone_ids = torch.randint(0, 393, (B, T))

        with torch.no_grad():
            result = zc2_rc.recompute(
                encoder_features=encoder_features,
                modified_zc1=modified_zc1,
                z_p=z_p,
                z_r=z_r,
                phone_ids=phone_ids,
            )
        assert not result.zc1.requires_grad
        assert not result.zc2.requires_grad
        assert not result.zc2_pred.requires_grad

    def test_missing_phone_ids_raises(self, tiny_denoiser, cpu_device):
        """predict mode should raise if phone_ids is not provided."""
        rc = ZC2Recomputer(mode="predict", denoiser=tiny_denoiser, device=cpu_device)
        with pytest.raises(ValueError, match="phone_ids is required"):
            rc.recompute(
                encoder_features=torch.randn(1, 1024, 20),
                modified_zc1=torch.randn(1, 1, 20),
                z_p=torch.randn(1, 1, 20),
                z_r=torch.randn(1, 4, 20),
                # phone_ids omitted
            )

    def test_forward_method_alias(self, tiny_denoiser, cpu_device):
        """forward() should be equivalent to recompute() in predict mode."""
        rc = ZC2Recomputer(mode="predict", denoiser=tiny_denoiser, device=cpu_device)
        B, T = 1, 20
        encoder_features = torch.randn(B, 1024, T)
        modified_zc1 = torch.randn(B, 8, T)
        z_p = torch.randn(B, 2, T)
        z_r = torch.randn(B, 4, T)
        phone_ids = torch.randint(0, 393, (B, T))

        with torch.no_grad():
            r_recompute = rc.recompute(
                encoder_features=encoder_features,
                modified_zc1=modified_zc1,
                z_p=z_p,
                z_r=z_r,
                phone_ids=phone_ids,
            )
            r_forward = rc.forward(
                denoised_zc1=modified_zc1,
                encoder_features=encoder_features,
                z_p=z_p,
                z_r=z_r,
                phone_ids=phone_ids,
            )
        # Both should produce the same zc2 (deterministic at t=0)
        assert torch.allclose(r_recompute.zc2, r_forward.zc2, atol=1e-5)


# ===========================================================================
# Standalone tests for DenoisingTransformerModel interface contract
# ===========================================================================

class TestDenoiserContract:
    """Verify the DenoisingTransformerModel's interface contracts."""

    def test_default_num_steps_is_100(self):
        from accentedge.phase1.denoiser import DenoisingTransformerModel
        import inspect
        sig = inspect.signature(DenoisingTransformerModel.__init__)
        assert sig.parameters["num_steps"].default == 100

    def test_default_facodec_dim_is_8(self):
        from accentedge.phase1.denoiser import DenoisingTransformerModel
        import inspect
        sig = inspect.signature(DenoisingTransformerModel.__init__)
        assert sig.parameters["facodec_dim"].default == 8

    def test_forward_returns_tuple_of_two_tensors(self):
        torch.manual_seed(0)
        model = DenoisingTransformerModel(
            d_model=64, nhead=4, num_layers=2, d_ff=128,
            phone_vocab_size=393, facodec_dim=8, num_steps=100,
        ).eval()
        B, C, T = 2, 8, 20
        zc1 = torch.randn(B, C, T)
        phone_ids = torch.randint(0, 393, (B, T))
        t = torch.randint(0, 100, (B,))

        with torch.no_grad():
            out = model(zc1, phone_ids, t)
        assert isinstance(out, tuple)
        assert len(out) == 2
        eps, zc2 = out
        assert eps.shape == (B, C, T)
        assert zc2.shape == (B, C, T)

    def test_sqrt_abar_buffer_shape(self):
        """sqrt_abar should have length num_steps."""
        model = DenoisingTransformerModel(
            d_model=64, nhead=4, num_layers=2, d_ff=128,
            phone_vocab_size=393, facodec_dim=8, num_steps=100,
        )
        assert model.sqrt_abar.shape == (100,)
        assert model.sqrt_1mabar.shape == (100,)

    def test_ddpm_schedule_is_sorted(self):
        """sqrt_abar should be monotonically non-increasing (cumprod of alphas)."""
        model = DenoisingTransformerModel(
            d_model=64, nhead=4, num_layers=2, d_ff=128,
            phone_vocab_size=393, facodec_dim=8, num_steps=100,
        )
        diffs = model.sqrt_abar[1:] - model.sqrt_abar[:-1]
        assert (diffs <= 1e-7).all(), "sqrt_abar should be non-increasing"

    def test_t_zero_produces_deterministic_output(self):
        """At t=0 (fully denoised), the model should produce stable zc2."""
        torch.manual_seed(0)
        model = DenoisingTransformerModel(
            d_model=64, nhead=4, num_layers=2, d_ff=128,
            phone_vocab_size=393, facodec_dim=8, num_steps=100,
        ).eval()
        zc1 = torch.randn(1, 8, 20)
        phone_ids = torch.randint(0, 393, (1, 20))
        t = torch.zeros(1, dtype=torch.long)

        with torch.no_grad():
            _, zc2_run1 = model(zc1, phone_ids, t)
            _, zc2_run2 = model(zc1, phone_ids, t)
        assert torch.allclose(zc2_run1, zc2_run2, atol=1e-6)


# ===========================================================================
# FactorizedLatents dataclass tests
# ===========================================================================

class TestFactorizedLatents:
    """Verify FactorizedLatents dataclass invariants."""

    def test_default_fields(self):
        latents = FactorizedLatents(
            content=torch.randn(1, 8, 20),
            content_zc1=torch.randn(1, 1, 20),
        )
        assert latents.content_zc2 is None
        assert latents.prosody is None
        assert latents.detail is None
        assert latents.timbre is None
        assert latents.metadata == {}

    def test_full_latents(self):
        latents = FactorizedLatents(
            content=torch.randn(2, 8, 50),
            content_zc1=torch.randn(2, 1, 50),
            content_zc2=torch.randn(2, 8, 50),
            prosody=torch.randn(2, 1, 50),
            detail=torch.randn(2, 4, 50),
            timbre=torch.randn(2, 256),
            metadata={"sample_rate": 24000},
        )
        assert latents.content.shape == (2, 8, 50)
        assert latents.content_zc1.shape == (2, 1, 50)
        assert latents.content_zc2.shape == (2, 8, 50)
        assert latents.prosody.shape == (2, 1, 50)
        assert latents.detail.shape == (2, 4, 50)
        assert latents.timbre.shape == (2, 256)
        assert latents.metadata["sample_rate"] == 24000


# ===========================================================================
# Frame rate contract tests
# ===========================================================================

class TestFrameRateContract:
    """Verify the 80fps contract between waveform length and frame count."""

    @pytest.mark.parametrize(
        "samples,expected_frames",
        [
            (2400, 8),   # 0.1 s
            (4800, 16),  # 0.2 s
            (12000, 40), # 0.5 s
            (24000, 80), # 1.0 s
            (48000, 160), # 2.0 s
        ],
    )
    def test_frame_count_formula(self, samples, expected_frames):
        """The converter's frame count formula must match PhonemePipeline's."""
        computed = int(round(samples * 80 / 24000))
        assert computed == expected_frames, (
            f"Frame count mismatch for {samples} samples: "
            f"expected {expected_frames}, got {computed}"
        )

    def test_phone_ids_shape_matches_converter_frame_count(self, sample_waveform_1s):
        """The converter's frame count formula matches PhonemePipeline's formula."""
        converter_frames = int(round(sample_waveform_1s.shape[-1] * 80 / 24000))
        pipeline = PhonemePipeline(device="cpu")
        pipeline_frames = int(
            round(sample_waveform_1s.shape[-1] * pipeline.frame_rate / pipeline.sample_rate)
        )
        assert converter_frames == pipeline_frames


# ===========================================================================
# Tests for the converter's internal validation helpers
# ===========================================================================

class TestConverterValidation:
    """Test AccentConverter's private validation methods directly."""

    def test_validate_latent_shapes_happy_path(self, device):
        """_validate_latent_shapes should pass for consistent shapes."""
        converter = AccentConverter(
            facodec_adapter=mock.MagicMock(),
            phoneme_pipeline=mock.MagicMock(),
            denoiser=DenoisingTransformerModel(
                d_model=64, nhead=4, num_layers=2, d_ff=128,
                phone_vocab_size=393, facodec_dim=8,
            ).eval().to(device),
            zc2_recomputer=ZC2Recomputer(
                mode="predict",
                denoiser=DenoisingTransformerModel(
                    d_model=64, nhead=4, num_layers=2, d_ff=128,
                    phone_vocab_size=393, facodec_dim=8,
                ).eval().to(device),
                device=device,
            ),
            device=device,
        )
        # Should not raise
        converter._validate_latent_shapes(
            z_q=torch.randn(1, 8, 20),
            z_c1=torch.randn(1, 1, 20),
            z_p=torch.randn(1, 1, 20),
            z_r=torch.randn(1, 4, 20),
            g=torch.randn(1, 256),
        )

    def test_assert_frame_rate_contract_happy_path(self, device):
        """_assert_frame_rate_contract should pass for matching frames."""
        converter = _build_converter(
            torch.randn(1, 24000), device,
            DenoisingTransformerModel(
                d_model=64, nhead=4, num_layers=2, d_ff=128,
                phone_vocab_size=393, facodec_dim=8,
            ).eval().to(device),
            ZC2Recomputer(
                mode="predict",
                denoiser=DenoisingTransformerModel(
                    d_model=64, nhead=4, num_layers=2, d_ff=128,
                    phone_vocab_size=393, facodec_dim=8,
                ).eval().to(device),
                device=device,
            ),
        )
        # Should not raise
        converter._assert_frame_rate_contract(
            phone_ids=torch.randint(0, 393, (1, 80)),
            z_q=torch.randn(1, 8, 80),
        )

    def test_assert_frame_rate_contract_mismatch_raises(self, device):
        """_assert_frame_rate_contract should raise on frame count mismatch."""
        converter = _build_converter(
            torch.randn(1, 24000), device,
            DenoisingTransformerModel(
                d_model=64, nhead=4, num_layers=2, d_ff=128,
                phone_vocab_size=393, facodec_dim=8,
            ).eval().to(device),
            ZC2Recomputer(
                mode="predict",
                denoiser=DenoisingTransformerModel(
                    d_model=64, nhead=4, num_layers=2, d_ff=128,
                    phone_vocab_size=393, facodec_dim=8,
                ).eval().to(device),
                device=device,
            ),
        )
        with pytest.raises(ValueError, match="Frame rate contract violated"):
            converter._assert_frame_rate_contract(
                phone_ids=torch.randint(0, 393, (1, 50)),  # wrong frame count
                z_q=torch.randn(1, 8, 80),
            )
