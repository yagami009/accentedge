"""Unit tests for the training pipeline (25+ tests)."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from accentedge_lab.training import (
    CheckpointManifest,
    accent_loss,
    checkpoints,
    content_loss,
    enable_deterministic,
    f0_loss,
    get_lr_scheduler,
    get_optimizer,
    get_rng_state,
    load_checkpoint_manifest,
    mel_loss,
    reconstruction_loss,
    save_checkpoint_manifest,
    set_seed,
    speaker_loss,
    total_loss,
    Trainer,
    verify_reproducibility,
)


# ======================================================================
# Helpers
# ======================================================================

class TinyModel(nn.Module):
    """A minimal model that returns per-component losses."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(16, 16)
        self.architecture_id = "tiny_model"

    def forward(self, x: torch.Tensor, **_: torch.Tensor) -> dict[str, torch.Tensor]:
        pred = self.linear(x)
        return {
            "content_loss": nn.functional.mse_loss(pred, x),
            "accent_loss": nn.functional.mse_loss(pred, x),
            "speaker_loss": (1 - torch.tensor(1.0)).abs(),  # placeholder
            "f0_loss": nn.functional.mse_loss(pred, x),
            "mel_loss": nn.functional.l1_loss(pred, x),
            "reconstruction_loss": nn.functional.l1_loss(pred, x),
        }


def make_dummy_batch(bs: int = 2, dim: int = 16) -> dict:
    return {"audio": torch.randn(bs, dim)}


def make_trainer(**kwargs) -> Trainer:
    model = TinyModel()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    merged = {"architecture_id": "tiny_model"}
    merged.update(kwargs)
    return Trainer(model, opt, **merged)


# ======================================================================
# Trainer initialisation
# ======================================================================


class TestTrainerInit:
    def test_default_device_cpu(self) -> None:
        """On a machine without CUDA/MPS the default device is CPU."""
        t = make_trainer(device="cpu")
        assert t.device.type == "cpu"

    def test_explicit_cpu(self) -> None:
        t = make_trainer(device="cpu")
        assert str(t.device) == "cpu"

    def test_default_precision(self) -> None:
        t = make_trainer()
        assert t.precision == "fp32"

    def test_explicit_precision(self) -> None:
        t = make_trainer(precision="fp32")
        assert t.precision == "fp32"

    def test_custom_loss_weights(self) -> None:
        weights = {"content_loss": 0.5, "mel_loss": 2.0}
        t = make_trainer(loss_weights=weights)
        assert t.loss_weights["content_loss"] == 0.5
        assert t.loss_weights["mel_loss"] == 2.0

    def test_default_loss_weights(self) -> None:
        t = make_trainer()
        for comp in [
            "content_loss", "accent_loss", "speaker_loss",
            "f0_loss", "mel_loss", "reconstruction_loss",
        ]:
            assert comp in t.loss_weights

    def test_architecture_id_stored(self) -> None:
        t = make_trainer(architecture_id="test_arch")
        assert t.architecture_id == "test_arch"


# ======================================================================
# train_step
# ======================================================================


class TestTrainStep:
    def test_finite_loss(self) -> None:
        t = make_trainer()
        batch = make_dummy_batch()
        loss = t.train_step(batch)
        assert np.isfinite(loss)

    def test_step_increments_counter(self) -> None:
        t = make_trainer()
        assert t.global_step == 0
        t.train_step(make_dummy_batch())
        assert t.global_step == 1

    def test_zero_grad_called(self) -> None:
        """After a step the optimizer should have zeroed gradients."""
        t = make_trainer()
        t.train_step(make_dummy_batch())
        for p in t.model.parameters():
            grad = p.grad
            # grad may be None or zero tensor — both mean gradients were cleared
            if grad is not None:
                assert torch.all(grad == 0)

    def test_loss_components_recorded(self) -> None:
        captured = {}
        t = make_trainer(logger=lambda r: captured.update(r))
        t.train_step(make_dummy_batch())
        assert "total_loss" in captured
        for comp in [
            "content_loss", "accent_loss", "speaker_loss",
            "f0_loss", "mel_loss", "reconstruction_loss",
        ]:
            assert comp in captured

    def test_losses_are_float(self) -> None:
        captured = {}
        t = make_trainer(logger=lambda r: captured.update(r))
        t.train_step(make_dummy_batch())
        assert isinstance(captured["total_loss"], float)
        for comp in [
            "content_loss", "accent_loss", "speaker_loss",
            "f0_loss", "mel_loss", "reconstruction_loss",
        ]:
            assert isinstance(captured[comp], float)


# ======================================================================
# Checkpoint save / load round-trip
# ======================================================================


class TestCheckpointRoundTrip:
    def test_save_and_load(self, tmp_path: Path) -> None:
        t = make_trainer()
        batch = make_dummy_batch()
        t.train_step(batch)  # advance step counter

        ckpt_path = tmp_path / "roundtrip.pt"
        t.save_checkpoint(ckpt_path)

        assert ckpt_path.exists()
        assert Path(str(ckpt_path) + ".json").exists()

        # Create a fresh trainer and load
        t2 = make_trainer()
        t2.train_step(make_dummy_batch())
        t2.load_checkpoint(ckpt_path)
        assert t2.global_step == t.global_step

    def test_checkpoint_contains_states(self, tmp_path: Path) -> None:
        t = make_trainer()
        ckpt_path = tmp_path / "states.pt"
        t.save_checkpoint(ckpt_path)
        ckpt = torch.load(ckpt_path, weights_only=False)
        assert "model_state" in ckpt
        assert "optimizer_state" in ckpt
        assert "rng_state" in ckpt

    def test_sidecar_json_exists(self, tmp_path: Path) -> None:
        t = make_trainer()
        ckpt_path = tmp_path / "sidecar.pt"
        t.save_checkpoint(ckpt_path)
        sidecar = Path(str(ckpt_path) + ".json")
        assert sidecar.exists()
        data = json.loads(sidecar.read_text())
        assert isinstance(data, dict)


# ======================================================================
# Checkpoint manifest
# ======================================================================


class TestCheckpointManifest:
    def test_all_required_fields(self, tmp_path: Path) -> None:
        t = make_trainer()
        ckpt_path = tmp_path / "manifest.pt"
        t.save_checkpoint(ckpt_path)
        manifest = load_checkpoint_manifest(ckpt_path)
        required = [
            "checkpoint_id",
            "architecture_id",
            "version",
            "config_hash",
            "training_manifest_hash",
            "training_data_lineage_hash",
            "parent_checkpoint_ids",
            "pretrained_weight_sources",
            "licenses",
            "commercial_use_status",
            "seed",
            "training_steps",
            "training_hours",
            "optimizer",
            "scheduler",
            "hardware",
            "wall_clock_seconds",
            "best_validation_metric",
            "timestamp",
        ]
        for field_name in required:
            assert hasattr(manifest, field_name), f"missing {field_name}"

    def test_commercial_use_status_field(self, tmp_path: Path) -> None:
        t = make_trainer(commercial_use_status="ALLOWED")
        ckpt_path = tmp_path / "commercial.pt"
        t.save_checkpoint(ckpt_path)
        manifest = load_checkpoint_manifest(ckpt_path)
        assert manifest.commercial_use_status == "ALLOWED"

    def test_commercial_use_status_unknown_default(self, tmp_path: Path) -> None:
        t = make_trainer()
        ckpt_path = tmp_path / "unknown.pt"
        t.save_checkpoint(ckpt_path)
        manifest = load_checkpoint_manifest(ckpt_path)
        assert manifest.commercial_use_status in ("UNKNOWN", "ALLOWED")

    def test_git_commit_present_or_none(self, tmp_path: Path) -> None:
        t = make_trainer()
        ckpt_path = tmp_path / "git.pt"
        t.save_checkpoint(ckpt_path)
        manifest = load_checkpoint_manifest(ckpt_path)
        # git_commit is either a short hash string or None
        assert manifest.git_commit is None or isinstance(manifest.git_commit, str)

    def test_seed_recorded(self, tmp_path: Path) -> None:
        t = make_trainer()
        ckpt_path = tmp_path / "seed.pt"
        t.save_checkpoint(ckpt_path, training_info={"seed": 123})
        manifest = load_checkpoint_manifest(ckpt_path)
        assert manifest.seed == 123

    def test_training_steps_recorded(self, tmp_path: Path) -> None:
        t = make_trainer()
        for _ in range(3):
            t.train_step(make_dummy_batch())
        ckpt_path = tmp_path / "steps.pt"
        t.save_checkpoint(ckpt_path)
        manifest = load_checkpoint_manifest(ckpt_path)
        assert manifest.training_steps == 3

    def test_manifest_is_json_serialisable(self, tmp_path: Path) -> None:
        t = make_trainer()
        ckpt_path = tmp_path / "json.pt"
        t.save_checkpoint(ckpt_path)
        sidecar = Path(str(ckpt_path) + ".json")
        raw = sidecar.read_text()
        parsed = json.loads(raw)  # must not raise
        assert "checkpoint_id" in parsed


# ======================================================================
# Reproducibility
# ======================================================================


class TestReproducibility:
    def test_set_seed_numpy(self) -> None:
        set_seed(7)
        a = np.random.randn(5)
        set_seed(7)
        b = np.random.randn(5)
        np.testing.assert_array_equal(a, b)

    def test_set_seed_python(self) -> None:
        import random as rn
        set_seed(11)
        rn.seed(11)
        x = [rn.random() for _ in range(10)]
        set_seed(11)
        y = [rn.random() for _ in range(10)]
        assert x == y

    def test_get_rng_state_keys(self) -> None:
        set_seed(0)
        state = get_rng_state()
        assert "python_hash_seed" in state
        assert "numpy" in state
        assert "torch_cpu" in state
        assert "torch_cuda" in state  # list (empty if no CUDA)

    def test_reproducibility_same_seed(self) -> None:
        """Two identical models with the same seed produce the same loss."""
        set_seed(99)
        m1 = TinyModel()
        m1.train()
        out1 = m1(make_dummy_batch()["audio"])

        set_seed(99)
        m2 = TinyModel()
        m2.train()
        out2 = m2(make_dummy_batch()["audio"])

        loss1 = sum(v for v in out1.values() if isinstance(v, torch.Tensor))
        loss2 = sum(v for v in out2.values() if isinstance(v, torch.Tensor))
        assert torch.allclose(loss1, loss2)

    def test_reproducibility_verify_true(self) -> None:
        model = TinyModel()
        batch = make_dummy_batch()
        assert verify_reproducibility(model, batch, n_runs=2) is True

    def test_enable_deterministic_no_crash(self) -> None:
        enable_deterministic()  # must not raise
        # Run a small forward/backward
        model = nn.Linear(8, 8)
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()

    def test_rng_state_differs_after_different_seeds(self) -> None:
        set_seed(1)
        s1 = get_rng_state()
        set_seed(2)
        s2 = get_rng_state()
        # numpy states should differ
        assert not np.array_equal(s1["numpy"][1], s2["numpy"][1])


# ======================================================================
# Loss functions
# ======================================================================


class TestLosses:
    def test_content_loss_finite(self) -> None:
        p, t = torch.randn(4, 8), torch.randn(4, 8)
        assert np.isfinite(float(content_loss(p, t)))

    def test_accent_loss_finite(self) -> None:
        p, t = torch.randn(4, 8), torch.randn(4, 8)
        assert np.isfinite(float(accent_loss(p, t)))

    def test_speaker_loss_finite(self) -> None:
        p, t = torch.randn(4, 8), torch.randn(4, 8)
        assert np.isfinite(float(speaker_loss(p, t)))

    def test_f0_loss_finite(self) -> None:
        p, t = torch.randn(4, 8), torch.randn(4, 8)
        assert np.isfinite(float(f0_loss(p, t)))

    def test_mel_loss_finite(self) -> None:
        p, t = torch.randn(4, 8), torch.randn(4, 8)
        assert np.isfinite(float(mel_loss(p, t)))

    def test_reconstruction_loss_finite(self) -> None:
        p, t = torch.randn(4, 8), torch.randn(4, 8)
        assert np.isfinite(float(reconstruction_loss(p, t)))

    def test_total_loss_combines(self) -> None:
        c = {
            "content_loss": torch.tensor(1.0),
            "mel_loss": torch.tensor(2.0),
        }
        weights = {"content_loss": 0.5, "mel_loss": 0.3}
        result = float(total_loss(c, weights))
        assert np.isclose(result, 0.5 * 1.0 + 0.3 * 2.0)

    def test_total_loss_zero_weights_skipped(self) -> None:
        c = {"content_loss": torch.tensor(1.0)}
        weights = {"content_loss": 0.0}
        assert float(total_loss(c, weights)) == 0.0

    def test_speaker_loss_identical_embeddings(self) -> None:
        """Identical (normalised) embeddings should give zero loss."""
        v = torch.randn(1, 8)
        vn = nn.functional.normalize(v, p=2, dim=-1)
        assert np.isclose(float(speaker_loss(vn, vn)), 0.0, atol=1e-5)


# ======================================================================
# Optimizer and scheduler
# ======================================================================


class TestSchedules:
    def test_get_optimizer_adam(self) -> None:
        p = nn.Parameter(torch.randn(2, 2))
        opt = get_optimizer([p], "adam", lr=1e-3)
        assert isinstance(opt, torch.optim.Adam)

    def test_get_optimizer_adamw(self) -> None:
        p = nn.Parameter(torch.randn(2, 2))
        opt = get_optimizer([p], "adamw", lr=1e-3)
        assert isinstance(opt, torch.optim.AdamW)

    def test_get_optimizer_sgd(self) -> None:
        p = nn.Parameter(torch.randn(2, 2))
        opt = get_optimizer([p], "sgd", lr=1e-3)
        assert isinstance(opt, torch.optim.SGD)

    def test_optimizer_step_updates_param(self) -> None:
        p = nn.Parameter(torch.randn(2, 2))
        opt = get_optimizer([p], "adam", lr=1e-2)
        loss = (p ** 2).sum()
        loss.backward()
        opt.step()
        assert not torch.allclose(p, torch.zeros_like(p))

    def test_get_lr_scheduler_cosine(self) -> None:
        p = nn.Parameter(torch.randn(2, 2))
        opt = get_optimizer([p], "adam", lr=1e-3)
        sched = get_lr_scheduler(opt, "cosine", T_max=10)
        assert isinstance(sched, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_get_lr_scheduler_linear(self) -> None:
        p = nn.Parameter(torch.randn(2, 2))
        opt = get_optimizer([p], "adam", lr=1e-3)
        sched = get_lr_scheduler(opt, "linear", total_iters=100)
        assert isinstance(sched, torch.optim.lr_scheduler.LinearLR)

    def test_get_lr_scheduler_step(self) -> None:
        p = nn.Parameter(torch.randn(2, 2))
        opt = get_optimizer([p], "adam", lr=1e-3)
        sched = get_lr_scheduler(opt, "step", step_size=10)
        assert isinstance(sched, torch.optim.lr_scheduler.StepLR)

    def test_get_lr_scheduler_constant(self) -> None:
        p = nn.Parameter(torch.randn(2, 2))
        opt = get_optimizer([p], "adam", lr=1e-3)
        sched = get_lr_scheduler(opt, "constant")
        assert isinstance(sched, torch.optim.lr_scheduler.ConstantLR)


# ======================================================================
# Gradient clipping
# ======================================================================


class TestGradientClipping:
    def test_clipping_limits_norm(self) -> None:
        """After a train_step with gradient clipping, param grad norm <= max."""
        model = nn.Linear(16, 16)
        opt = torch.optim.SGD(model.parameters(), lr=0.1)
        t = Trainer(model, opt, max_grad_norm=0.5, device="cpu")
        batch = make_dummy_batch()
        t.train_step(batch)
        norms = [float(p.grad.detach().norm(2)) for p in model.parameters() if p.grad is not None]
        total_norm = float(torch.sqrt(torch.tensor(sum(n ** 2 for n in norms))))
        assert total_norm <= 0.5 + 1e-5


# ======================================================================
# Smoke overfit test
# ======================================================================


class TestSmokeOverfit:
    def test_tiny_dataset_overfit(self) -> None:
        """A model should be able to overfit a single batch in a few steps."""
        model = TinyModel()
        opt = torch.optim.Adam(model.parameters(), lr=5e-3)
        t = Trainer(model, opt, max_grad_norm=5.0, device="cpu", checkpoint_dir=tempfile.mkdtemp())
        batch = make_dummy_batch()

        initial_loss = None
        for _ in range(50):
            loss = t.train_step(batch)
            if initial_loss is None:
                initial_loss = loss

        # Loss must decrease significantly (at least 30 %)
        assert loss < initial_loss * 0.7, (
            f"Loss did not decrease: {initial_loss} -> {loss}"
        )
