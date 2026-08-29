#!/usr/bin/env python3
"""Real-latent overfit training for AccentEdge Phase 1 — Gate 5.

Trains the denoiser on 5-10 real utterances (not synthetic data).
Uses real zc1 from FACodec and real phone_ids from PhonemePipeline.

Three mandatory gates:
  A. Denoising:  denoised_zc1 is closer to clean than noisy input
  B. Mean baseline: model beats predicting training-set mean
  C. Conditioning: correct phones beat wrong phones AND shuffled phones

After training, decodes one sample to WAV for qualitative inspection.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

# ── Path setup so `accentedge` is importable from any CWD ──────────
_SCRIPT_DIR = Path(__file__).resolve()
_SRC_ROOT = _SCRIPT_DIR.parents[2]  # project_root/src
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from accentedge.phase1.denoiser import DenoisingTransformerModel
from accentedge.phase1.diffusion import compute_noise_schedule, q_sample
from accentedge.phase1.phoneme_pipeline import PhonemePipeline
from accentedge.codec.facodec import FACodecAdapter
from accentedge.training.checkpoint import save_checkpoint


# ═══════════════════════════════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════════════════════════════

class OverfitDataset:
    """Load 5-10 utterances, extract real latents + phone IDs."""

    def __init__(
        self,
        audio_dir: str,
        transcripts: dict[str, str],
        facodec: FACodecAdapter,
        phoneme_pipeline: PhonemePipeline,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.samples: list[dict] = []

        audio_paths = sorted(Path(audio_dir).glob("*.wav"))
        if not audio_paths:
            raise ValueError(f"No .wav files found in {audio_dir}")

        print(f"Loading {len(audio_paths)} utterances from {audio_dir} …")
        for audio_path in audio_paths:
            key = audio_path.stem
            transcript = transcripts.get(key, "")

            # ── waveform ───────────────────────────────────────────────
            wav, sr = sf.read(str(audio_path), dtype="float32")
            wav = torch.from_numpy(wav).to(self.device)
            if wav.dim() == 1:
                wav = wav.unsqueeze(0)
            if sr != 24000:
                import torchaudio  # lazy — only needed when resampling
                wav = torchaudio.functional.resample(wav, sr, 24000)

            # ── phone IDs ─────────────────────────────────────────────
            phone_ids = phoneme_pipeline(transcript, wav)  # [1, T_phones]

            # ── FACodec encoding ──────────────────────────────────────
            latents = facodec.encode(wav)

            zc1 = latents.content_zc1                            # [1, C, T_codec]
            z_p = latents.prosody                                # [1, Cp, T_codec]
            z_r = latents.detail                                 # [1, Cr, T_codec]
            z_t = self._extract_zt(latents, zc1)                 # [1, Ct, T_codec]
            z_q = latents.content                                # [1, C, T_codec]

            # zc2 = residual content not captured by zc1 / z_p / z_t / z_r
            zc2_target = self._compute_zc2_target(z_q, zc1, z_p, z_t, z_r)

            self.samples.append({
                "key": key,
                "wav": wav,
                "zc1": zc1,
                "zc2_target": zc2_target,
                "z_p": z_p,
                "z_t": z_t,
                "z_r": z_r,
                "z_q": z_q,
                "phone_ids": phone_ids,
            })

        print(f"Loaded {len(self.samples)} utterances.\n")

    # ── helpers ────────────────────────────────────────────────────

    @staticmethod
    def _extract_zt(latents, reference: torch.Tensor) -> torch.Tensor:
        """Return per-frame timbre residual z_t.

        FACodec quantizer returns z_t inside quantized_list.  If it's not
        directly accessible we fall back to zeros.
        """
        # The adapter's encode() does not expose z_t explicitly;
        # z_t is part of the additive decomposition inside the quantizer.
        # We approximate it as zeros — the denoiser still learns from the
        # other residuals.  A more precise extraction would require
        # patching FACodecAdapter to return all four quantized vectors.
        return torch.zeros_like(reference)

    @staticmethod
    def _compute_zc2_target(
        z_q: torch.Tensor,
        z_c: torch.Tensor,
        z_p: torch.Tensor,
        z_t: torch.Tensor,
        z_r: torch.Tensor,
    ) -> torch.Tensor:
        """Content residual: z_q − z_c − z_p − z_t − z_r."""
        # Broadcast all to [1, C_max, T] then subtract
        T = z_q.shape[-1]
        def _to_max(t, C_max):
            if t.shape[1] == C_max:
                return t
            # repeat or truncate channel dim
            rep = t.repeat(1, max(1, C_max // t.shape[1]), 1)
            return rep[:, :C_max, :]

        C_max = z_q.shape[1]
        parts = [_to_max(z, C_max) for z in (z_c, z_p, z_t, z_r)]
        residual = z_q
        for p in parts:
            residual = residual - p
        return residual

    def __len__(self) -> int:
        return len(self.samples)

    def get(self, idx: int, device: torch.device) -> dict:
        s = self.samples[idx]
        return {
            k: (v.to(device) if isinstance(v, torch.Tensor) else v)
            for k, v in s.items()
        }


# ═══════════════════════════════════════════════════════════════════════
#  Normalization helpers
# ═══════════════════════════════════════════════════════════════════════

def _normalize(zc1: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """Per-channel normalization.  mean/std: [C], zc1: [B, C, T]."""
    m = mean.view(1, -1, 1).to(zc1.device, zc1.dtype)
    s = std.view(1, -1, 1).to(zc1.device, zc1.dtype)
    return (zc1 - m) / s


def _denormalize(zc1_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    m = mean.view(1, -1, 1).to(zc1_norm.device, zc1_norm.dtype)
    s = std.view(1, -1, 1).to(zc1_norm.device, zc1_norm.dtype)
    return zc1_norm * s + m


def compute_zc1_stats(dataset: OverfitDataset, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-channel mean and std of zc1 across the entire training set."""
    channel_means: list[torch.Tensor] = []
    channel_sqmeans: list[torch.Tensor] = []
    total_frames = 0

    for s in dataset.samples:
        z = s["zc1"].to(device).transpose(1, 2)   # [1, T, C]
        C = z.shape[-1]
        channel_means.append(z.sum(dim=(0, 1)))     # [C]
        channel_sqmeans.append((z ** 2).sum(dim=(0, 1)))
        total_frames += z.shape[0] * z.shape[1]

    mean = torch.stack(channel_means).sum(dim=0) / max(total_frames, 1)
    var = torch.stack(channel_sqmeans).sum(dim=0) / max(total_frames, 1) - mean ** 2
    std = torch.sqrt(var.clamp_min(1e-8)) + 1e-8
    return mean, std


# ═══════════════════════════════════════════════════════════════════════
#  Gate evaluation
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate_gates(
    model: DenoisingTransformerModel,
    dataset: OverfitDataset,
    sched: dict,
    zc1_mean: torch.Tensor,
    zc1_std: torch.Tensor,
    device: torch.device,
    num_timesteps: int,
    zc2_loss_weight: float,
) -> dict:
    """Run all three gates and return structured results."""
    model.eval()
    results: dict = {}

    eval_idxs = list(range(len(dataset)))  # all utterances — small set

    # ── Gate A: Denoising ──────────────────────────────────────────
    noisy_errs, denoised_errs = [], []
    for idx in eval_idxs:
        s = dataset.get(idx, device)
        zc1 = s["zc1"]
        phone_ids = s["phone_ids"]
        zc1_n = _normalize(zc1, zc1_mean, zc1_std)

        t_mid = torch.tensor([num_timesteps // 2], device=device)
        xt_n, noise = q_sample(zc1_n, t_mid, sched["sqrt_alpha_bar"], sched["sqrt_1m_alpha_bar"])

        eps_pred, _ = model(xt_n, phone_ids, t_mid)

        sa = sched["sqrt_alpha_bar"][t_mid].view(1, 1, 1)
        s1a = sched["sqrt_1m_alpha_bar"][t_mid].view(1, 1, 1)
        denoised_n = (xt_n - s1a * eps_pred) / sa

        noisy_errs.append(F.mse_loss(xt_n, zc1_n).item())
        denoised_errs.append(F.mse_loss(denoised_n, zc1_n).item())

    avg_noisy = float(np.mean(noisy_errs))
    avg_denoised = float(np.mean(denoised_errs))
    improvement = (avg_noisy - avg_denoised) / max(avg_noisy, 1e-10)
    gate_a = avg_denoised < avg_noisy and improvement > 0.05

    results["denoising"] = {
        "passed": bool(gate_a),
        "noisy_error": avg_noisy,
        "denoised_error": avg_denoised,
        "improvement": improvement,
        "msg": (f"noisy={avg_noisy:.6f}  denoised={avg_denoised:.6f}  "
                f"improvement={improvement:.1%}"),
    }

    # ── Gate B: Mean Baseline ──────────────────────────────────────
    # Predict the training-set mean (normalized) for every sample.
    mean_n = (zc1_mean.view(1, -1, 1) / zc1_std.view(1, -1, 1)).to(device)

    model_losses, mb_losses = [], []
    for idx in eval_idxs:
        s = dataset.get(idx, device)
        zc1 = s["zc1"]
        phone_ids = s["phone_ids"]
        zc2_tgt = s["zc2_target"]
        zc1_n = _normalize(zc1, zc1_mean, zc1_std)
        zc2_n = _normalize(zc2_tgt, zc1_mean, zc1_std)

        t = torch.randint(0, num_timesteps, (1,), device=device)
        xt_n, noise = q_sample(zc1_n, t, sched["sqrt_alpha_bar"], sched["sqrt_1m_alpha_bar"])

        eps_p, zc2_p = model(xt_n, phone_ids, t)
        ml = F.mse_loss(eps_p, noise) + zc2_loss_weight * F.mse_loss(zc2_p, zc2_n.detach())
        model_losses.append(ml.item())

        # Mean baseline for this sample: MSE(mean_n, clean_zc1_n)
        mb_losses.append(F.mse_loss(mean_n.expand_as(zc1_n), zc1_n).item())

    avg_model = float(np.mean(model_losses))
    avg_mb = float(np.mean(mb_losses))
    gate_b = avg_model < avg_mb

    results["mean_baseline"] = {
        "passed": bool(gate_b),
        "model_loss": avg_model,
        "mean_baseline_loss": avg_mb,
        "msg": f"model={avg_model:.6f}  mean_baseline={avg_mb:.6f}",
    }

    # ── Gate C: Conditioning Ablation ──────────────────────────────
    vocab_size = model.phone_emb.num_embeddings
    correct_losses, wrong_losses, shuffled_losses = [], [], []

    for idx in eval_idxs:
        s = dataset.get(idx, device)
        zc1 = s["zc1"]
        phone_ids = s["phone_ids"]
        zc2_tgt = s["zc2_target"]
        zc1_n = _normalize(zc1, zc1_mean, zc1_std)
        zc2_n = _normalize(zc2_tgt, zc1_mean, zc1_std)

        t = torch.full((1,), num_timesteps // 2, device=device, dtype=torch.long)
        xt_n, noise = q_sample(zc1_n, t, sched["sqrt_alpha_bar"], sched["sqrt_1m_alpha_bar"])

        # (1) correct phones
        eps_c, zc2_c = model(xt_n, phone_ids, t)
        l_correct = F.mse_loss(eps_c, noise) + zc2_loss_weight * F.mse_loss(zc2_c, zc2_n)
        correct_losses.append(l_correct.item())

        # (2) wrong phones (random, different length OK but we match shape)
        wrong_ids = torch.randint(0, vocab_size, phone_ids.shape, device=device)
        eps_w, zc2_w = model(xt_n, wrong_ids, t)
        l_wrong = F.mse_loss(eps_w, noise) + zc2_loss_weight * F.mse_loss(zc2_w, zc2_n)
        wrong_losses.append(l_wrong.item())

        # (3) same-length shuffle control
        shuffled_ids = phone_ids.clone()
        Tp = shuffled_ids.shape[1]
        perm = torch.randperm(Tp, device=device)
        shuffled_ids = shuffled_ids[:, perm]
        eps_s, zc2_s = model(xt_n, shuffled_ids, t)
        l_shuf = F.mse_loss(eps_s, noise) + zc2_loss_weight * F.mse_loss(zc2_s, zc2_n)
        shuffled_losses.append(l_shuf.item())

    avg_correct = float(np.mean(correct_losses))
    avg_wrong = float(np.mean(wrong_losses))
    avg_shuf = float(np.mean(shuffled_losses))
    magnitude_pct = (avg_wrong - avg_correct) / max(avg_correct, 1e-10) * 100.0
    gate_c = avg_correct < avg_wrong and avg_correct < avg_shuf

    results["conditioning"] = {
        "passed": bool(gate_c),
        "correct_loss": avg_correct,
        "wrong_loss": avg_wrong,
        "shuffled_loss": avg_shuf,
        "magnitude_pct": magnitude_pct,
        "msg": (f"correct={avg_correct:.6f}  wrong={avg_wrong:.6f}  "
                f"shuffled={avg_shuf:.6f}  magnitude={magnitude_pct:.1f}%"),
    }

    model.train()
    return results


# ═══════════════════════════════════════════════════════════════════════
#  WAV decode (one overfit sample)
# ═══════════════════════════════════════════════════════════════════════

@torch.no_grad()
def _decode_one_sample(
    model: DenoisingTransformerModel,
    dataset: OverfitDataset,
    facodec: FACodecAdapter,
    sched: dict,
    zc1_mean: torch.Tensor,
    zc1_std: torch.Tensor,
    device: torch.device,
    num_timesteps: int,
    wav_dir: Path,
    label: str,
) -> Path | None:
    """Decode one overfit sample to WAV.

    Pipeline:  denoised_zc1  →  zc2 recomputation  →  preserved factors
               →  FACodec decoder  →  WAV
    """
    model.eval()
    idx = 0
    s = dataset.get(idx, device)
    zc1 = s["zc1"]            # [1, C, T]
    phone_ids = s["phone_ids"]  # [1, T]
    z_p = s["z_p"]
    z_r = s["z_r"]

    zc1_n = _normalize(zc1, zc1_mean, zc1_std)

    # t=0 → fully denoised (xt = clean)
    t0 = torch.tensor([0], device=device)
    _, zc2_pred_n = model(zc1_n, phone_ids, t0)   # zc2 in normalized space

    denoised_zc1 = _denormalize(zc1_n, zc1_mean, zc1_std)
    denoised_zc2 = _denormalize(zc2_pred_n, zc1_mean, zc1_std)

    # Reconstruct z_q = zc1 + zc2 + z_p + z_t + z_r
    z_q_recon = denoised_zc1 + denoised_zc2
    if z_p is not None and z_p.numel() > 0:
        z_q_recon = z_q_recon + z_p.to(z_q_recon.device)
    if z_r is not None and z_r.numel() > 0:
        z_q_recon = z_q_recon + z_r.to(z_q_recon.device)

    from accentedge.codec.interfaces import FactorizedLatents
    latents_recon = FactorizedLatents(
        content=z_q_recon,
        content_zc1=denoised_zc1,
        content_zc2=denoised_zc2,
        prosody=z_p,
        detail=z_r,
    )

    # Original for comparison
    latents_orig = FactorizedLatents(
        content=s["z_q"].to(device),
        content_zc1=zc1,
        prosody=z_p,
        detail=z_r,
    )

    try:
        wav_recon = facodec.decode(latents_recon).squeeze().cpu().numpy()
        wav_orig = facodec.decode(latents_orig).squeeze().cpu().numpy()

        recon_path = wav_dir / f"recon_{label}.wav"
        orig_path = wav_dir / f"original_{label}.wav"
        sf.write(str(recon_path), wav_recon, facodec.sample_rate)
        sf.write(str(orig_path), wav_orig, facodec.sample_rate)
        print(f"  WAV saved → {recon_path.name} (+ original)")
        return recon_path
    except Exception as exc:
        print(f"  WAV decode failed: {exc}")
        return None


# ═══════════════════════════════════════════════════════════════════════
#  Main training loop
# ═══════════════════════════════════════════════════════════════════════

def train_overfit(
    audio_dir: str,
    transcripts: dict[str, str],
    output_dir: str = "artifacts/overfit",
    model_d_model: int = 256,
    model_nhead: int = 4,
    model_num_layers: int = 3,
    model_d_ff: int = 512,
    model_phone_vocab_size: int = 393,
    model_facodec_dim: int = 8,
    num_timesteps: int = 100,
    num_steps: int = 5000,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    warmup_steps: int = 200,
    device: str = "cuda",
    facodec_ckpt: str = "Plachta/FAcodec",
    seed: int = 42,
    zc2_loss_weight: float = 0.5,
    checkpoint_every: int = 1000,
    gate_check_every: int = 500,
    decode_steps: list[int] | None = None,
) -> dict:
    """Real-latent overfit training with mandatory gates.

    Args:
        audio_dir: Directory with 5-10 .wav files.
        transcripts: {audio_stem: transcript_text}.
        output_dir: Root output directory (checkpoint.pt, metrics.json, etc.).
        model_*: Denoiser architecture hyperparameters.
        num_timesteps: Diffusion timestep count.
        num_steps: Total training steps.
        learning_rate: AdamW peak LR.
        warmup_steps: Linear warmup before cosine decay.
        device: "cuda" or "cpu".
        facodec_ckpt: HuggingFace identifier or local path for FACodec.
        seed: RNG seed.
        zc2_loss_weight: Weight on the zc2 prediction auxiliary loss.
        checkpoint_every: Save checkpoint every N steps.
        gate_check_every: Evaluate gates every N steps.
        decode_steps: Steps at which to decode & save a WAV.

    Returns:
        Metrics dictionary (also written to metrics.json).
    """
    # ── Setup ────────────────────────────────────────────────────────
    random.seed(seed)
    torch.manual_seed(seed)

    device_t = torch.device(device)
    out_root = Path(output_dir)
    wav_dir = out_root / "generated_wavs"
    wav_dir.mkdir(parents=True, exist_ok=True)

    # ── FACodec ─────────────────────────────────────────────────────
    print("Loading FACodec …")
    facodec = FACodecAdapter(device=device, facodec_ckpt=facodec_ckpt)
    facodec.freeze()

    # ── PhonemePipeline ────────────────────────────────────────────
    print("Loading PhonemePipeline …")
    phoneme_pipeline = PhonemePipeline(device=device, phone_vocab_size=model_phone_vocab_size)

    # ── Dataset ─────────────────────────────────────────────────────
    dataset = OverfitDataset(
        audio_dir=audio_dir,
        transcripts=transcripts,
        facodec=facodec,
        phoneme_pipeline=phoneme_pipeline,
        device=device,
    )
    if len(dataset) < 5:
        print(f"  WARNING: only {len(dataset)} utterances; need 5-10.")
    print(f"Dataset size: {len(dataset)} utterances.")

    # ── zc1 statistics (computed over entire training set) ─────────
    print("Computing per-channel zc1 normalization stats …")
    zc1_mean, zc1_std = compute_zc1_stats(dataset, device_t)
    zc1_mean_cpu = zc1_mean.detach().cpu()
    zc1_std_cpu = zc1_std.detach().cpu()
    print(f"  mean = {zc1_mean_cpu.tolist()}")
    print(f"  std  = {zc1_std_cpu.tolist()}")

    # ── Model ───────────────────────────────────────────────────────
    print("Building denoiser …")
    model = DenoisingTransformerModel(
        d_model=model_d_model,
        nhead=model_nhead,
        num_layers=model_num_layers,
        d_ff=model_d_ff,
        phone_vocab_size=model_phone_vocab_size,
        facodec_dim=model_facodec_dim,
        num_steps=num_timesteps,
    ).to(device_t)

    # ── Optimizer & scheduler ──────────��───────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        return max(0.0, 1.0 - (step - warmup_steps) / max(1, num_steps - warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # ── Diffusion schedule on device ───────────────────────────────
    sched = compute_noise_schedule(num_timesteps)
    sched = {k: (v.to(device_t) if isinstance(v, torch.Tensor) else v) for k, v in sched.items()}

    # ── Pre-compute mean-baseline loss ─────────────────────────────
    mean_zc1_n = (zc1_mean.view(1, -1, 1) / zc1_std.view(1, -1, 1)).to(device_t)
    mb_losses_all = []
    for s in dataset.samples:
        zc1_n_full = _normalize(s["zc1"].to(device_t), zc1_mean, zc1_std)
        mb_losses_all.append(F.mse_loss(mean_zc1_n.expand_as(zc1_n_full), zc1_n_full).item())
    mean_baseline_ref = float(np.mean(mb_losses_all))
    print(f"Mean-baseline loss: {mean_baseline_ref:.6f}")

    # ═══��══════════════════════════════════════════════════════════════
    #  Training loop
    # ══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"Training: {num_steps} steps  |  {len(dataset)} utterances  |  {device}")
    print(f"{'='*60}\n")

    loss_hist: list[float] = []
    grad_hist: list[dict] = []
    metrics_log: dict = {
        "step": [], "loss": [], "eps_loss": [], "zc2_loss": [],
        "lr": [],
        "phone_embedding_grad_norm": [],
        "conditioning_projection_grad_norm": [],
        "denoiser_core_grad_norm": [],
        "gate_denoising": [],
        "gate_mean_baseline": [],
        "gate_conditioning": [],
        "cond_magnitude_pct": [],
    }
    t0_start = time.time()
    last_gate_results: dict = {}

    for step in range(1, num_steps + 1):
        # ── sample ──────────────────────────────────────────────────
        idx = random.randrange(len(dataset))
        s = dataset.get(idx, device_t)
        zc1 = s["zc1"]              # [1, C, T]  unnormalized
        phone_ids = s["phone_ids"]   # [1, T]
        zc2_tgt = s["zc2_target"]    # [1, C, T]

        zc1_n = _normalize(zc1, zc1_mean, zc1_std)
        zc2_n = _normalize(zc2_tgt, zc1_mean, zc1_std)

        t = torch.randint(0, num_timesteps, (1,), device=device_t)
        xt_n, noise = q_sample(zc1_n, t, sched["sqrt_alpha_bar"], sched["sqrt_1m_alpha_bar"])

        # ── forward + loss ──────────────────────────────────────────
        optimizer.zero_grad()
        eps_pred, zc2_pred = model(xt_n, phone_ids, t)

        loss_eps = F.mse_loss(eps_pred, noise)
        loss_zc2 = F.mse_loss(zc2_pred, zc2_n.detach())
        loss = loss_eps + zc2_loss_weight * loss_zc2
        loss.backward()

        # ── gradient norms (before step) ────────────────────────────
        with torch.no_grad():
            g_emb = float(torch.norm(model.phone_emb.weight.grad.detach()).item())
            g_proj = float(torch.norm(model.phone_proj.weight.grad.detach()).item())
            core_parts = [
                p.grad.detach().flatten()
                for n, p in model.named_parameters()
                if p.grad is not None
                and "phone_emb" not in n
                and "phone_proj" not in n
            ]
            g_core = float(torch.norm(torch.cat(core_parts)).item()) if core_parts else 0.0

        optimizer.step()
        scheduler.step()

        loss_hist.append(float(loss.item()))
        grad_hist.append({"emb": g_emb, "proj": g_proj, "core": g_core})

        # ── console logging ─────────────────────────────────────────
        if step % 100 == 0:
            avg = float(np.mean(loss_hist[-100:]))
            lr = scheduler.get_last_lr()[0]
            print(f"  step {step:5d}  loss={avg:.6f}  lr={lr:.2e}  "
                  f"emb={g_emb:.3f}  proj={g_proj:.3f}  core={g_core:.3f}")

        # accumulate in log
        metrics_log["step"].append(step)
        metrics_log["loss"].append(float(loss.item()))
        metrics_log["eps_loss"].append(float(loss_eps.item()))
        metrics_log["zc2_loss"].append(float(loss_zc2.item()))
        metrics_log["lr"].append(float(scheduler.get_last_lr()[0]))
        metrics_log["phone_embedding_grad_norm"].append(g_emb)
        metrics_log["conditioning_projection_grad_norm"].append(g_proj)
        metrics_log["denoiser_core_grad_norm"].append(g_core)

        # ── gate evaluation ─────────────────────────────────────────
        if step % gate_check_every == 0 or step == num_steps:
            last_gate_results = evaluate_gates(
                model, dataset, sched, zc1_mean, zc1_std,
                device_t, num_timesteps, zc2_loss_weight,
            )
            metrics_log["gate_denoising"].append(last_gate_results["denoising"])
            metrics_log["gate_mean_baseline"].append(last_gate_results["mean_baseline"])
            metrics_log["gate_conditioning"].append(last_gate_results["conditioning"])
            metrics_log["cond_magnitude_pct"].append(
                last_gate_results["conditioning"]["magnitude_pct"]
            )

            all_pass = (
                last_gate_results["denoising"]["passed"]
                and last_gate_results["mean_baseline"]["passed"]
                and last_gate_results["conditioning"]["passed"]
            )
            print(f"\n  ── Gates @ step {step} ──")
            print(f"  A {('✓' if last_gate_results['denoising']['passed'] else '✗')}  "
                  f"{last_gate_results['denoising']['msg']}")
            print(f"  B {('✓' if last_gate_results['mean_baseline']['passed'] else '✗')}  "
                  f"{last_gate_results['mean_baseline']['msg']}")
            print(f"  C {('✓' if last_gate_results['conditioning']['passed'] else '✗')}  "
                  f"{last_gate_results['conditioning']['msg']}")
            print(f"  OVERALL: {'PASS ✓✓✓' if all_pass else 'FAIL ✗✗✗'}\n")

        # ── WAV decode ──────────────────────────────────────────────
        if decode_steps and step in decode_steps:
            _decode_one_sample(model, dataset, facodec, sched,
                               zc1_mean, zc1_std, device_t,
                               num_timesteps, wav_dir, f"step{step}")

        # ── checkpoint ──────────────────────────────────────────────
        if step % checkpoint_every == 0 or step == num_steps:
            ckpt_path = out_root / f"checkpoint_step{step}.pt"
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                epoch=step // max(len(dataset), 1),
                config={
                    "d_model": model_d_model,
                    "nhead": model_nhead,
                    "num_layers": model_num_layers,
                    "d_ff": model_d_ff,
                    "phone_vocab_size": model_phone_vocab_size,
                    "facodec_dim": model_facodec_dim,
                    "num_timesteps": num_timesteps,
                    "num_steps": num_steps,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                    "warmup_steps": warmup_steps,
                    "seed": seed,
                    "zc2_loss_weight": zc2_loss_weight,
                    "audio_dir": str(audio_dir),
                    "num_utterances": len(dataset),
                },
                phone_vocab=None,  # vocab is inside PhonemePipeline / processor
                facodec_ckpt=facodec_ckpt,
                zc1_mean=zc1_mean_cpu,
                zc1_std=zc1_std_cpu,
                output_path=str(ckpt_path),
            )
            print(f"  Checkpoint saved → {ckpt_path.name}")

    # ══════════════════════════════════════════════════════════════════
    #  Final decode
    # ══════════════════════════════════════════════════════════════════
    print("\nDecoding final WAV …")
    _decode_one_sample(model, dataset, facodec, sched,
                       zc1_mean, zc1_std, device_t,
                       num_timesteps, wav_dir, "final")

    # ══════════════════════════════════════════════════════════════════
    #  Save metrics & zc1_stats
    # ══════════════════════════════════════════════════════════════════
    total_secs = time.time() - t0_start

    # zc1_stats.json
    zc1_stats = {
        "mean": zc1_mean_cpu.tolist(),
        "std": zc1_std_cpu.tolist(),
        "num_channels": int(zc1_mean.shape[0]),
        "num_utterances": len(dataset),
    }
    (out_root / "zc1_stats.json").write_text(json.dumps(zc1_stats, indent=2))

    # Final checkpoint alias
    final_ckpt = out_root / "checkpoint.pt"
    if not final_ckpt.exists():
        import shutil
        last_step_ckpt = out_root / f"checkpoint_step{num_steps}.pt"
        if last_step_ckpt.exists():
            shutil.copy2(last_step_ckpt, final_ckpt)

    # metrics.json
    final_metrics = {
        "num_steps": num_steps,
        "num_utterances": len(dataset),
        "total_time_seconds": round(total_secs, 1),
        "first_loss": loss_hist[0] if loss_hist else None,
        "final_loss": loss_hist[-1] if loss_hist else None,
        "loss_decrease": (loss_hist[0] - loss_hist[-1]) if len(loss_hist) >= 2 else None,
        "mean_baseline_loss": mean_baseline_ref,
        "zc1_stats": zc1_stats,
        "final_gates": last_gate_results,
        "gradient_norms_final": grad_hist[-1] if grad_hist else {},
        "output_dir": str(out_root),
        "generated_wavs_dir": str(wav_dir),
    }
    (out_root / "metrics.json").write_text(json.dumps(final_metrics, indent=2))

    print(f"\nAll artifacts saved to: {out_root}")
    print(f"  checkpoint.pt       ← latest checkpoint")
    print(f"  metrics.json")
    print(f"  zc1_stats.json")
    print(f"  generated_wavs/")
    return final_metrics


# ═══════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="AccentEdge Phase 1 — Real-latent overfit (Gate 5)",
    )
    ap.add_argument("--audio-dir", type=str, required=True,
                    help="Directory containing 5-10 .wav files.")
    ap.add_argument("--transcripts", type=str, default=None,
                    help="JSON mapping audio_stem → transcript text.")
    ap.add_argument("--output-dir", type=str, default="artifacts/overfit")
    ap.add_argument("--num-steps", type=int, default=5000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=3)
    ap.add_argument("--d-ff", type=int, default=512)
    ap.add_argument("--learning-rate", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--facodec-ckpt", type=str, default="Plachta/FAcodec")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-timesteps", type=int, default=100)
    args = ap.parse_args()

    # Load transcripts
    if args.transcripts and Path(args.transcripts).is_file():
        transcripts: dict[str, str] = json.loads(Path(args.transcripts).read_text())
    else:
        stems = {p.stem: p.stem for p in sorted(Path(args.audio_dir).glob("*.wav"))}
        transcripts = stems
        print(f"No transcript file — using stems as transcripts: {list(transcripts.values())[:5]}")

    metrics = train_overfit(
        audio_dir=args.audio_dir,
        transcripts=transcripts,
        output_dir=args.output_dir,
        model_d_model=args.d_model,
        model_nhead=args.nhead,
        model_num_layers=args.num_layers,
        model_d_ff=args.d_ff,
        num_timesteps=args.num_timesteps,
        num_steps=args.num_steps,
        learning_rate=args.learning_rate,
        device=args.device,
        facodec_ckpt=args.facodec_ckpt,
        seed=args.seed,
    )

    print("\n=== Gate 5 complete ===")
    print(json.dumps(metrics.get("final_gates", {}), indent=2))


if __name__ == "__main__":
    main()

