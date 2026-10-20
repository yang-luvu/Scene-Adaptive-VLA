import math
import shutil
from pathlib import Path
from typing import Any, Optional

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn


class FixedCheckpoint(Callback):
    """
    Save a rolling checkpoint to a fixed path.

    mode="latest" overwrites on a fixed global-step interval.
    mode="best" overwrites when the monitored training loss improves.
    """

    def __init__(
        self,
        mode: str,
        dirpath: str = "./checkpoints",
        filename: str = "latest.ckpt",
        monitor: str = "train/loss",
        step_interval: int = 3000,
        enabled: bool = True,
    ) -> None:
        super().__init__()
        if mode not in ("latest", "best"):
            raise ValueError(f"mode must be 'latest' or 'best', got {mode}")
        if int(step_interval) <= 0:
            raise ValueError(f"step_interval must be > 0, got {step_interval}")

        self.mode = str(mode)
        self.dirpath = Path(dirpath)
        self.filename = str(filename)
        self.monitor = str(monitor)
        self.step_interval = int(step_interval)
        self.enabled = bool(enabled)

        self.best_value: Optional[float] = None
        self.last_seen_global_step: int = -1
        self._missing_loss_warned: bool = False

    @property
    def checkpoint_path(self) -> Path:
        return self.dirpath / self.filename

    def _barrier(self, trainer: pl.Trainer) -> None:
        strategy = getattr(trainer, "strategy", None)
        if strategy is not None and hasattr(strategy, "barrier"):
            strategy.barrier()

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

    def _should_save_latest(self, global_step: int) -> bool:
        return global_step > 0 and (global_step % self.step_interval) == 0

    def _maybe_save_best(self, trainer: pl.Trainer, outputs: Any, global_step: int) -> None:
        loss_tensor = self._extract_loss_tensor(outputs=outputs, trainer=trainer)
        if loss_tensor is None:
            if trainer.is_global_zero and not self._missing_loss_warned:
                rank_zero_warn(
                    f"[FixedCheckpoint] could not read loss from outputs['loss'] or '{self.monitor}'."
                )
                self._missing_loss_warned = True
            return

        loss_value = self._sync_loss_value(trainer=trainer, loss_tensor=loss_tensor)
        if not math.isfinite(loss_value):
            if trainer.is_global_zero:
                rank_zero_warn(
                    f"[FixedCheckpoint] got non-finite loss={loss_value} at step={global_step}, skip this step."
                )
            return

        if self.best_value is not None and loss_value >= self.best_value:
            return

        self.best_value = loss_value
        self._overwrite_checkpoint(trainer=trainer)
        if trainer.is_global_zero:
            rank_zero_info(
                "[FixedCheckpoint] saved best checkpoint "
                f"to {self.checkpoint_path} at step={global_step}, "
                f"best_{self.monitor}={loss_value:.6f}."
            )

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

        if self.mode == "latest":
            if not self._should_save_latest(global_step):
                return
            self._overwrite_checkpoint(trainer=trainer)
            if trainer.is_global_zero:
                rank_zero_info(
                    "[FixedCheckpoint] saved latest checkpoint "
                    f"to {self.checkpoint_path} at step={global_step}."
                )
            return

        self._maybe_save_best(trainer=trainer, outputs=outputs, global_step=global_step)
