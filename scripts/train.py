"""Train a diffusion policy (or baseline) on generated trajectory data.

Usage:
  python scripts/train.py --model diffusion_policy --tasks microsphere,yeast
  python scripts/train.py --model diffusion_policy --tasks all
  python scripts/train.py --model diffusion_policy --resume checkpoints/epoch_0100.pt
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse

import torch
from torch.utils.data import DataLoader, random_split

from src.data.augmentation import TrajectoryAugmentation
from src.data.dataset import TrajectoryDataset
from src.training.trainer import Trainer
from src.utils.config import (
    DiffusionPolicyConfig,
    TrainingConfig,
    load_diffusion_policy_config,
    load_training_config,
)
from src.vla.diffusion_policy import DiffusionPolicy
from src.vla.text_encoder import CachedTextEmbeddings, TextEncoder


def get_text_embeddings(task_names: list[str], cache_path: Path | None = None) -> dict:
    """Get text embeddings for tasks, precomputing if needed."""
    if cache_path and cache_path.exists():
        cached = CachedTextEmbeddings(str(cache_path))
        return {name: cached.get(name) for name in task_names if name in cached}

    encoder = TextEncoder()
    embeddings = encoder.precompute_all()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings, cache_path)
    return {k: v for k, v in embeddings.items() if k in task_names}


def main():
    parser = argparse.ArgumentParser(description="Train a VLA policy.")
    parser.add_argument("--model", type=str, default="diffusion_policy",
                        choices=["diffusion_policy", "bc", "act"])
    parser.add_argument("--tasks", type=str, default="all",
                        help="Comma-separated task names or 'all'")
    parser.add_argument("--data-root", type=str, default="data/trajectories",
                        help="Path to trajectory data")
    parser.add_argument("--model-config", type=str, default="configs/model/diffusion_policy.yaml")
    parser.add_argument("--training-config", type=str, default="configs/training/default.yaml")
    parser.add_argument("--checkpoint-dir", type=str, default="data/checkpoints")
    parser.add_argument("--text-cache", type=str, default="data/text_embeddings.pt")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--auto-resume", action="store_true", help="Auto-resume from latest checkpoint if exists")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-dir", type=str, default=None,
                        help="TensorBoard log directory (overrides config)")
    parser.add_argument("--precompute-text", action="store_true",
                        help="Precompute and cache text embeddings, then exit")
    parser.add_argument("--img-size", type=int, default=224,
                        help="Image size for training (should match simulator image_size)")
    args = parser.parse_args()

    ALL_TASKS = ["embryo", "oocyte", "real_sperm_head", "whole_sperm", "microsphere"]
    if args.tasks == "all":
        task_names = ALL_TASKS
    else:
        task_names = [t.strip() for t in args.tasks.split(",")]

    # Precompute text embeddings
    if args.precompute_text:
        get_text_embeddings(task_names, Path(args.text_cache))
        print(f"Text embeddings cached to {args.text_cache}")
        return

    text_embeddings = get_text_embeddings(task_names, Path(args.text_cache))
    print(f"Task embeddings: {list(text_embeddings.keys())}")

    # Load configs
    if args.model == "diffusion_policy":
        model_config = load_diffusion_policy_config(Path(args.model_config))
    else:
        raise NotImplementedError(f"Model {args.model} not yet implemented")

    train_config = load_training_config(Path(args.training_config))
    if args.log_dir:
        train_config.log_dir = args.log_dir

    # Create model
    if args.model == "diffusion_policy":
        model = DiffusionPolicy(
            obs_horizon=model_config.obs_horizon,
            pred_horizon=model_config.pred_horizon,
            action_dim=model_config.action_dim,
            visual_backbone=model_config.visual_backbone,
            visual_output_dim=model_config.visual_output_dim,
            text_dim=model_config.text_dim,
            condition_hidden_dims=model_config.condition_hidden_dims,
            condition_output_dim=model_config.condition_output_dim,
            unet_dims=model_config.unet_dims,
            num_diffusion_steps=model_config.num_diffusion_steps,
            num_ddim_steps=model_config.num_ddim_steps,
            beta_schedule=model_config.beta_schedule,
        )
    else:
        raise NotImplementedError()

    print(f"Model: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

    # Create dataset
    dataset = TrajectoryDataset(
        data_root=Path(args.data_root),
        task_names=task_names,
        text_embeddings=text_embeddings,
        obs_horizon=model_config.obs_horizon,
        pred_horizon=model_config.pred_horizon,
        img_size=(args.img_size, args.img_size),
        transform=TrajectoryAugmentation(),
    )

    if len(dataset) == 0:
        print("No data found! Run scripts/generate_data.py first.")
        return

    print(f"Dataset: {len(dataset)} samples")

    # Split
    val_size = int(len(dataset) * train_config.val_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=train_config.batch_size,
        shuffle=True, num_workers=args.num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_config.batch_size,
        shuffle=False, num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        checkpoint_dir=Path(args.checkpoint_dir),
    )

    # Resume from checkpoint
    resume_path = None
    if args.resume:
        resume_path = Path(args.resume)
    elif args.auto_resume:
        # Find latest checkpoint (prefer best.pt, then latest epoch)
        ckpt_dir = Path(args.checkpoint_dir)
        best_ckpt = ckpt_dir / "best.pt"
        if best_ckpt.exists():
            resume_path = best_ckpt
        else:
            # Find latest epoch checkpoint
            epoch_ckpts = sorted(ckpt_dir.glob("epoch_*.pt"), reverse=True)
            if epoch_ckpts:
                resume_path = epoch_ckpts[0]

    if resume_path and resume_path.exists():
        trainer.load_checkpoint(resume_path)
        print(f"Resumed from {resume_path} (epoch {trainer.current_epoch})")
    elif args.auto_resume:
        print("No checkpoint found, starting from scratch")

    history = trainer.train()
    print(f"Training complete. Best val loss: {trainer.best_val_loss:.6f}")


if __name__ == "__main__":
    main()
