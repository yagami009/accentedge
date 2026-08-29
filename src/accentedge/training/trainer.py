"""Generic Trainer abstraction for AccentEdge candidates."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Iterator

import torch

from accentedge.models.training.losses import total_loss
from accentedge.models.training.reproducibility import (
    enable_deterministic,
    get_rng_state,
    set_seed,
)
from accentedge.models.training.checkpoints import save_checkpoint_manifest


LOSS_COMPONENTS = (
    "content_loss",
    "accent_loss",
    "speaker_loss",
    "f0_loss",
    "mel_loss",
    "reconstruction_loss",
)


class Trainer:
    """Device-aware, mixed-precision training loop with checkpointing."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        loss_weights: dict[str, float] | None = None,
        device: str | None = None,
        precision: str = "fp32",
        max_grad_norm: float = 1.0,
        checkpoint_dir: str | Path = "checkpoints",
        architecture_id: str = "",
        version: str = "0.0.0",
        training_data_lineage_hash: str = "",
        licenses: list[str] | None = None,
        commercial_use_status: str = "UNKNOWN",
        logger: Callable[[dict], None] | None = None,
    ) -> None:
        self.architecture_id = architecture_id
        self.version = version

        # Device resolution
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.precision = precision
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.loss_weights = loss_weights or {
            "content_loss": 1.0,
            "accent_loss": 1.0,
            "speaker_loss": 1.0,
            "f0_loss": 1.0,
            "mel_loss": 1.0,
            "reconstruction_loss": 1.0,
        }
        self.max_grad_norm = max_grad_norm
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or self._default_logger

        self.global_step = 0
        self.best_val_metric: float | None = None
        self._start_wall_time: float | None = None

        # AMP scaler (fp16 only)
        self._scaler = (
            torch.amp.GradScaler("cuda") if precision == "fp16" and self.device.type == "cuda" else None
        )

        # Provenance fields for manifests
        self._training_data_lineage_hash = training_data_lineage_hash
        self._licenses = licenses or ["Proprietary"]
        self._commercial_use_status = commercial_use_status

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    @staticmethod
    def _default_logger(record: dict) -> None:
        print(
            f"[step {record.get('step', '?')}] "
            f"loss={record.get('total_loss', float('nan')):.4f}  "
            + " ".join(
                f"{k}={record.get(k, float('nan')):.4f}"
                for k in LOSS_COMPONENTS
                if k in record
            )
        )

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    def set_seed(self, seed: int) -> None:
        """Set all RNG seeds for reproducibility."""
        set_seed(seed)
        self._rng_state_at_start = get_rng_state()

    # ------------------------------------------------------------------
    # Forward / loss helper
    # ------------------------------------------------------------------
    def _compute_loss(self, batch: dict[str, Any]) -> tuple[torch.Tensor, dict]:
        """Run a forward pass and return (total_loss, component_dict)."""
        model_out = self.model(
            batch.get("audio", batch.get("input")),
            **{
                k: v
                for k, v in batch.items()
                if k not in ("audio", "input")
            },
        )

        components: dict[str, torch.Tensor] = {}
        if isinstance(model_out, dict):
            for comp in LOSS_COMPONENTS:
                if comp in model_out:
                    components[comp] = model_out[comp]

            if "total_loss" in model_out:
                return model_out["total_loss"], components

            return total_loss(components, self.loss_weights), components
        else:
            # Plain tensor output — treat as content_loss
            return model_out, {"content_loss": model_out.detach()}

    # ------------------------------------------------------------------
    # Core training step
    # ------------------------------------------------------------------
    def train_step(self, batch: dict[str, Any]) -> float:
        """Execute one training step. Returns the scalar loss."""
        self.model.train()
        batch = self._move_to_device(batch)

        if self.precision == "fp16" and self.device.type == "cuda":
            with torch.autocast("cuda", dtype=torch.float16):
                loss, components = self._compute_loss(batch)
        else:
            loss, components = self._compute_loss(batch)

        loss = loss.sum() if loss.ndim > 0 else loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        record: dict[str, Any] = {
            "step": self.global_step,
            "total_loss": float(loss.detach().cpu()),
        }
        for k, v in components.items():
            tv = v.detach().cpu()
            record[k] = float(tv) if tv.numel() == 1 else float(tv.sum())

        self.logger(record)
        return float(loss.detach().cpu())

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def validate(
        self, val_loader: Iterator[dict[str, Any]]
    ) -> dict[str, float]:
        """Run validation and return average metrics."""
        self.model.eval()
        accum: dict[str, float] = {}
        n_batches = 0

        for batch in val_loader:
            batch = self._move_to_device(batch)
            loss, components = self._compute_loss(batch)
            accum["val_loss"] = accum.get("val_loss", 0.0) + float(loss.cpu())
            for k, v in components.items():
                accum[f"val_{k}"] = accum.get(f"val_{k}", 0.0) + float(v.cpu())
            n_batches += 1

        if n_batches == 0:
            return {"val_loss": float("inf")}

        avg = {k: v / n_batches for k, v in accum.items()}
        return avg

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, path: str | Path, training_info: dict | None = None) -> Path:
        """Persist model + optimizer state and write manifest sidecar."""
        base = Path(path)
        ckpt = {
            "global_step": self.global_step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "rng_state": get_rng_state(),
            "architecture_id": self.architecture_id,
            "version": self.version,
        }
        if self.scheduler is not None:
            ckpt["scheduler_state"] = self.scheduler.state_dict()

        torch.save(ckpt, base)

        # Sidecar manifest
        info = {
            "architecture_id": self.architecture_id,
            "version": self.version,
            "seed": 42,
            "optimizer": type(self.optimizer).__name__,
            "scheduler": type(self.scheduler).__name__
            if self.scheduler
            else "None",
            "training_steps": self.global_step,
            "training_data_lineage_hash": self._training_data_lineage_hash,
            "licenses": self._licenses,
            "commercial_use_status": self._commercial_use_status,
        }
        if training_info:
            info.update(training_info)

        save_checkpoint_manifest(
            self.model,
            config=info,
            training_info=info,
            path=base,
        )
        return base

    def load_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Restore model / optimizer from a checkpoint path."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.optimizer.load_state_dict(ckpt["optimizer_state"])
        self.global_step = ckpt.get("global_step", 0)
        if self.scheduler is not None and "scheduler_state" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state"])
        return ckpt

    # ------------------------------------------------------------------
    # Fit loop
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: Iterator[dict[str, Any]],
        val_loader: Iterator[dict[str, Any]] | None = None,
        max_steps: int = 50_000,
        checkpoint_every: int = 1_000,
        validation_every: int = 500,
    ) -> None:
        """Main training loop."""
        self._start_wall_time = time.time()
        self.set_seed(42)
        self.model.train()

        while self.global_step < max_steps:
            for batch in train_loader:
                if self.global_step >= max_steps:
                    break

                loss = self.train_step(batch)

                if val_loader and self.global_step % validation_every == 0:
                    metrics = self.validate(val_loader)
                    val_loss = metrics.get("val_loss", float("nan"))
                    self.logger(
                        {"step": self.global_step, "val_loss": val_loss}
                    )
                    if (
                        self.best_val_metric is None
                        or val_loss < self.best_val_metric
                    ):
                        self.best_val_metric = val_loss
                        self.save_checkpoint(
                            self.checkpoint_dir / "best.pt",
                            {"best_validation_metric": val_loss},
                        )

                if self.global_step % checkpoint_every == 0:
                    self.save_checkpoint(
                        self.checkpoint_dir / f"step_{self.global_step:07d}.pt"
                    )

        elapsed_h = (time.time() - self._start_wall_time) / 3600 if self._start_wall_time else 0.0

        # Final checkpoint with full training_info
        final_info = {
            "training_steps": self.global_step,
            "training_hours": elapsed_h,
            "wall_clock_seconds": time.time() - self._start_wall_time
            if self._start_wall_time
            else 0.0,
            "best_validation_metric": self.best_val_metric,
        }
        self.save_checkpoint(
            self.checkpoint_dir / "final.pt", training_info=final_info
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            else:
                result[k] = v
        return result

