import math
import shutil
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn


class LossPlateauCheckpoint(Callback):
    """
    Save a rolling checkpoint when monitored training loss has not improved
    for a fixed number of optimizer steps.

    The checkpoint path is fixed and overwritten to avoid disk growth.
    """

    def __init__(
        self,
        patience_steps: int = 10_000,
        min_delta: float = 1e-4,
        monitor: str = "train/loss",
        dirpath: str = "./checkpoints",
        filename: str = "loss_plateau.ckpt",
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if int(patience_steps) <= 0:
            raise ValueError(f"patience_steps must be > 0, got {patience_steps}")
        if float(min_delta) < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {min_delta}")

        self.enabled = bool(enabled)
        self.patience_steps = int(patience_steps)
        self.min_delta = float(min_delta)
        self.monitor = str(monitor)
        self.dirpath = Path(dirpath)
        self.filename = str(filename)

        self.best_loss: Optional[float] = None
        self.bad_steps: int = 0
        self.last_seen_global_step: int = -1
        self._missing_loss_warned: bool = False

    @property
    def checkpoint_path(self) -> Path:
        return self.dirpath / self.filename

    def _barrier(self, trainer: pl.Trainer) -> None:
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None and hasattr(strategy, "barrier"):
            strategy.barrier()

    def _extract_loss_tensor(self, outputs: Any, trainer: pl.Trainer) -> Optional[torch.Tensor]:
        if isinstance(outputs, dict):
            output_loss = outputs.get("loss", None)
            if torch.is_tensor(output_loss):
                return output_loss.detach()

        metric_loss = trainer.callback_metrics.get(self.monitor, None)
        if torch.is_tensor(metric_loss):
            return metric_loss.detach()

        return None

    def _sync_loss_value(self, trainer: pl.Trainer, loss_tensor: torch.Tensor) -> float:
        strategy = getattr(trainer, "strategy", None)
        reduced = loss_tensor
        if strategy is not None and hasattr(strategy, "reduce"):
            reduced = strategy.reduce(loss_tensor, reduce_op="mean")
        return float(reduced.float().cpu().item())

    def _overwrite_checkpoint(self, trainer: pl.Trainer) -> None:
        ckpt_path = self.checkpoint_path
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)

        self._barrier(trainer)
        if trainer.is_global_zero and ckpt_path.exists():
            if ckpt_path.is_dir():
                shutil.rmtree(ckpt_path)
            else:
                ckpt_path.unlink()
        self._barrier(trainer)

        trainer.save_checkpoint(str(ckpt_path))
        self._barrier(trainer)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        del pl_module, batch, batch_idx

        if not self.enabled or trainer.sanity_checking:
            return

        global_step = int(trainer.global_step)
        if global_step <= 0:
            return
        if global_step <= self.last_seen_global_step:
            return
        self.last_seen_global_step = global_step

        loss_tensor = self._extract_loss_tensor(outputs=outputs, trainer=trainer)
        if loss_tensor is None:
            if trainer.is_global_zero:
                if not self._missing_loss_warned:
                    rank_zero_warn(
                    f"[LossPlateauCheckpoint] could not read loss from outputs['loss'] or '{self.monitor}'."
                    )
                    self._missing_loss_warned = True
            return
        loss_value = self._sync_loss_value(trainer=trainer, loss_tensor=loss_tensor)
        if not math.isfinite(loss_value):
            if trainer.is_global_zero:
                rank_zero_warn(
                    f"[LossPlateauCheckpoint] got non-finite loss={loss_value} at step={global_step}, skip this step."
                )
            return

        improved = self.best_loss is None or (loss_value < (self.best_loss - self.min_delta))
        if improved:
            self.best_loss = loss_value
            self.bad_steps = 0
            return

        self.bad_steps += 1
        if self.bad_steps < self.patience_steps:
            return

        self._overwrite_checkpoint(trainer=trainer)
        if trainer.is_global_zero:
            rank_zero_info(
                "[LossPlateauCheckpoint] saved rolling checkpoint "
                f"to {self.checkpoint_path} at step={global_step}, "
                f"current_loss={loss_value:.6f}, best_loss={self.best_loss:.6f}, "
                f"patience_steps={self.patience_steps}."
            )
        self.bad_steps = 0
