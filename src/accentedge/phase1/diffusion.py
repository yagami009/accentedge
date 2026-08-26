"""Phase 1 -- Diffusion math: noise schedule, sampling, strength mapping."""
from __future__ import annotations

import torch


def compute_noise_schedule(num_steps: int = 100, noise_min: float = 1e-4, noise_max: float = 2e-2):
    betas = torch.linspace(noise_min, noise_max, num_steps)
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas, "alphas": alphas, "alpha_bar": alpha_bar,
        "sqrt_alpha_bar": torch.sqrt(alpha_bar),
        "sqrt_1m_alpha_bar": torch.sqrt(1.0 - alpha_bar),
    }


def q_sample(x0, t, sqrt_alpha_bar, sqrt_1m_alpha_bar):
    noise = torch.randn_like(x0)
    xt = sqrt_alpha_bar[t].view(-1, 1, 1) * x0 + sqrt_1m_alpha_bar[t].view(-1, 1, 1) * noise
    return xt, noise


def ddim_step(x_t, eps_pred, t, t_prev, sqrt_alpha_bar, sqrt_1m_alpha_bar, eta: float = 0.0):
    bsz = x_t.size(0)
    sqrt_a_t = sqrt_alpha_bar[t].view(-1, 1, 1)
    sqrt_1ma_t = sqrt_1m_alpha_bar[t].view(-1, 1, 1)
    x0_pred = (x_t - sqrt_1ma_t * eps_pred) / sqrt_a_t
    x0_pred = torch.clamp(x0_pred, -10, 10)
    if t_prev is None:
        return x0_pred
    t_prev_int = int(t_prev) if not isinstance(t_prev, int) else t_prev
    if t_prev_int < 0:
        return x0_pred
    sqrt_a_prev = sqrt_alpha_bar[t_prev].view(-1, 1, 1)
    sqrt_1ma_prev = sqrt_1m_alpha_bar[t_prev].view(-1, 1, 1)
    noise = torch.randn_like(x_t)
    x_t = sqrt_a_prev * x0_pred + sqrt_1ma_prev * eps_pred + eta * sqrt_1ma_prev * noise
    return x_t


def strength_to_t_start(strength: float, num_steps: int = 100) -> int:
    strength = max(0.0, min(1.0, strength))
    return int(round(strength * num_steps))


def t_start_to_strength(t_start: int, num_steps: int = 100) -> float:
    return t_start / max(1, num_steps)
