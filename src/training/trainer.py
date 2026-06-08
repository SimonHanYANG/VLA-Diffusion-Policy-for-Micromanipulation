import math
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.utils.config import TrainingConfig
from src.utils.logger import setup_logger


class Trainer:
    """Generic trainer for Diffusion Policy, BC, and ACT models."""

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.checkpoint_dir = checkpoint_dir or Path("data/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        run_dir = Path(config.log_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=str(run_dir))

        self.logger = setup_logger("trainer")

        self.optimizer = AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        self.scheduler = self._build_scheduler()
        self.current_epoch = 0
        self.best_val_loss = float("inf")
        self.patience_counter = 0

        # History
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []

    def _build_scheduler(self) -> LambdaLR:
        """Cosine decay with linear warmup."""
        warmup = self.config.warmup_epochs
        total = self.config.num_epochs

        def lr_lambda(epoch: int) -> float:
            if epoch < warmup:
                return epoch / max(1, warmup)
            progress = (epoch - warmup) / max(1, total - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(self.optimizer, lr_lambda)

    def train(self) -> Dict[str, list]:
        """Run full training loop."""
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Train samples: {len(self.train_loader.dataset)}")
        if self.val_loader:
            self.logger.info(f"Val samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            start_time = time.time()

            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)

            val_loss = None
            if self.val_loader and epoch % self.config.val_freq_epochs == 0:
                val_loss = self._validate()
                self.val_losses.append(val_loss)

                # Early stopping
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.patience_counter = 0
                    self.save_checkpoint("best.pt")
                else:
                    self.patience_counter += 1

            # Periodic checkpoint
            if epoch % self.config.save_freq_epochs == 0 and epoch > 0:
                self.save_checkpoint(f"epoch_{epoch:04d}.pt")

            elapsed = time.time() - start_time
            lr = self.scheduler.get_last_lr()[0]

            self.writer.add_scalar("train/loss", train_loss, epoch)
            self.writer.add_scalar("train/lr", lr, epoch)
            if val_loss is not None:
                self.writer.add_scalar("val/loss", val_loss, epoch)

            log_msg = f"Epoch {epoch:4d} | loss: {train_loss:.6f} | lr: {lr:.2e} | time: {elapsed:.1f}s"
            if val_loss is not None:
                log_msg += f" | val_loss: {val_loss:.6f}"
            self.logger.info(log_msg)

            if self.patience_counter >= self.config.early_stopping_patience:
                self.logger.info(f"Early stopping at epoch {epoch}")
                break

            self.scheduler.step()

        self.logger.info(f"Training finished. Best val loss: {self.best_val_loss:.6f}")
        self.writer.close()
        return {"train_losses": self.train_losses, "val_losses": self.val_losses}

    def _train_epoch(self) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Train epoch {self.current_epoch}", leave=False)
        for batch_idx, batch in enumerate(pbar):
            obs_seq = batch["obs_seq"].to(self.device)
            action_seq = batch["action_seq"].to(self.device)
            text_emb = batch["text_emb"].to(self.device)

            loss = self.model.compute_loss(obs_seq, action_seq, text_emb)
            loss = loss / self.config.grad_accum_steps
            loss.backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0 or (batch_idx + 1) == len(self.train_loader):
                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.grad_accum_steps
            num_batches += 1
            pbar.set_postfix({"loss": f"{loss.item() * self.config.grad_accum_steps:.4f}"})

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def _validate(self) -> float:
        """Run validation. Returns average loss."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        for batch in tqdm(self.val_loader, desc="Validation", leave=False):
            obs_seq = batch["obs_seq"].to(self.device)
            action_seq = batch["action_seq"].to(self.device)
            text_emb = batch["text_emb"].to(self.device)

            loss = self.model.compute_loss(obs_seq, action_seq, text_emb)
            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def save_checkpoint(self, filename: str) -> None:
        """Save model and optimizer state."""
        path = self.checkpoint_dir / filename
        torch.save({
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_val_loss": self.best_val_loss,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "config": self.config,
        }, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: Path) -> None:
        """Load model and optimizer state."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            self.scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        self.current_epoch = ckpt["epoch"] + 1
        self.best_val_loss = ckpt["best_val_loss"]
        self.train_losses = ckpt.get("train_losses", [])
        self.val_losses = ckpt.get("val_losses", [])
        self.logger.info(f"Checkpoint loaded: {path} (epoch {ckpt['epoch']})")
