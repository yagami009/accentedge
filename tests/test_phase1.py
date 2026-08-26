"""Phase 1 tests."""
import torch
import pytest


class TestDiffusion:
    def test_noise_schedule(self):
        from accentedge.phase1.diffusion import compute_noise_schedule
        s = compute_noise_schedule(100)
        assert s["betas"].shape == (100,)
        assert s["alpha_bar"].shape == (100,)

    def test_q_sample(self):
        from accentedge.phase1.diffusion import q_sample, compute_noise_schedule
        s = compute_noise_schedule(100)
        x0 = torch.randn(2, 8, 20)
        t = torch.tensor([10, 50])
        xt, noise = q_sample(x0, t, s["sqrt_alpha_bar"], s["sqrt_1m_alpha_bar"])
        assert xt.shape == x0.shape

    def test_ddim_step(self):
        from accentedge.phase1.diffusion import ddim_step, compute_noise_schedule
        s = compute_noise_schedule(100)
        x = torch.randn(2, 8, 20)
        eps = torch.randn(2, 8, 20)
        out = ddim_step(x, eps, 10, 5, s["sqrt_alpha_bar"], s["sqrt_1m_alpha_bar"])
        assert out.shape == x.shape


class TestStrength:
    def test_mapping(self):
        from accentedge.phase1.strength import strength_to_t_start, t_start_to_strength
        assert strength_to_t_start(0.0) == 0
        assert strength_to_t_start(1.0) == 100
        assert strength_to_t_start(0.5) == 50
        assert t_start_to_strength(50) == 0.5

    def test_scheduler(self):
        from accentedge.phase1.strength import StrengthScheduler
        s = StrengthScheduler(num_steps=100)
        assert s(0.0) == 0
        assert s(1.0) == 100
        assert s(0.5) == 50


class TestDenoiser:
    def test_sinusoidal_pos_emb(self):
        from accentedge.phase1.denoiser import SinusoidalPosEmb
        pe = SinusoidalPosEmb(64)
        t = torch.tensor([0, 50, 99], dtype=torch.float32)
        out = pe(t)
        assert out.shape == (3, 64)

    def test_cond_layer_norm(self):
        from accentedge.phase1.denoiser import CondLayerNorm
        ln = CondLayerNorm(64)
        x = torch.randn(2, 10, 64)
        cond = torch.randn(2, 10, 128)
        out = ln(x, cond)
        assert out.shape == (2, 10, 64)

    def test_conv_ff(self):
        from accentedge.phase1.denoiser import ConvFeedForward
        ff = ConvFeedForward(d_model=64, d_ff=128)
        x = torch.randn(2, 64, 10)
        out = ff(x)
        assert out.shape == (2, 64, 10)

    def test_transformer_layer(self):
        from accentedge.phase1.denoiser import CustomTransformerEncoderLayer
        layer = CustomTransformerEncoderLayer(d_model=64, nhead=4, d_ff=128)
        x = torch.randn(2, 10, 64)
        cond = torch.randn(2, 10, 128)
        out = layer(x, cond)
        assert out.shape == (2, 10, 64)

    def test_transformer_encoder(self):
        from accentedge.phase1.denoiser import CustomTransformerEncoder
        enc = CustomTransformerEncoder(num_layers=2, d_model=64, nhead=4, d_ff=128)
        x = torch.randn(2, 10, 64)
        cond = torch.randn(2, 10, 128)
        out = enc(x, cond)
        assert out.shape == (2, 10, 64)

    def test_denoising_model_forward(self):
        from accentedge.phase1.denoiser import DenoisingTransformerModel
        model = DenoisingTransformerModel(
            d_model=64, nhead=4, num_layers=2, d_ff=128,
            phone_vocab_size=393, facodec_dim=8
        )
        zc1 = torch.randn(2, 8, 20)
        phone_ids = torch.randint(0, 392, (2, 20))
        t = torch.randint(0, 100, (2,))
        eps, zc2 = model(zc1, phone_ids, t)
        assert eps.shape == (2, 8, 20)
        assert zc2.shape == (2, 8, 20)


class TestCodecInterface:
    def test_interfaces(self):
        from accentedge.codec.interfaces import FactorizedLatents, FactorizedSpeechCodec
        latents = FactorizedLatents(zc1=torch.randn(2, 8, 20), zc2=torch.randn(2, 8, 20))
        assert latents.zc1.shape == (2, 8, 20)
