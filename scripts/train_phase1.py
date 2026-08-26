#!/usr/bin/env python3
"""Phase 1 denoiser training smoke test for Colab."""
import subprocess, sys, os

os.chdir("/content/accentedge")
os.environ["PYTHONPATH"] = "/content/FAC-FACodec:/content/accentedge/src:" + os.environ.get("PYTHONPATH", "")
sys.path.insert(0, "/content/accentedge/src")

import torch
import torch.nn.functional as F

from accentedge.phase1.denoiser import DenoisingTransformerModel
from accentedge.phase1.diffusion import compute_noise_schedule, q_sample

DEVICE = "cuda"
print(f"Device: {DEVICE}")
print(f"Torch: {torch.__version__}")

# Small model for smoke test
model = DenoisingTransformerModel(
    d_model=64, nhead=4, num_layers=2, d_ff=128,
    phone_vocab_size=393, facodec_dim=8
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
sched = compute_noise_schedule(100)
for k in sched:
    if isinstance(sched[k], torch.Tensor):
        sched[k] = sched[k].to(DEVICE)

# Check FACodec availability
print("\nChecking FAC-FACodec:")
import importlib.util
spec = importlib.util.find_spec("FACodec_AC")
print(f"  FACodec_AC module: {'found' if spec else 'NOT FOUND'}")

print("\n=== Smoke training (50 steps) ===")
losses = []
for step in range(50):
    zc1 = torch.randn(4, 8, 20, device=DEVICE)
    phone_ids = torch.randint(0, 392, (4, 20), device=DEVICE)
    t = torch.randint(0, 100, (4,), device=DEVICE)

    xt, noise = q_sample(zc1, t, sched["sqrt_alpha_bar"], sched["sqrt_1m_alpha_bar"])
    xt = xt.clone().detach().requires_grad_(True)

    optimizer.zero_grad()
    eps_pred, zc2_pred = model(xt, phone_ids, t)
    loss = F.mse_loss(eps_pred, noise)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

    if step % 10 == 0:
        print(f"  step {step:3d} | loss={loss.item():.6f}")

print(f"\nFirst loss: {losses[0]:.6f}")
print(f"Last  loss: {losses[-1]:.6f}")
print(f"Decrease: {losses[0] - losses[-1]:.6f}")
passed = losses[0] > losses[-1] * 1.1
print(f"\n{'PASS' if passed else 'FAIL'}: loss {'decreased' if passed else 'did NOT decrease'}")
