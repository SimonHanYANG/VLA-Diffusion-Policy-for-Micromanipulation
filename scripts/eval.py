"""Evaluate a trained model in the simulator.

Usage:
  python scripts/eval.py --checkpoint data/checkpoints/best.pt --task microsphere --episodes 100
  python scripts/eval.py --checkpoint data/checkpoints/best.pt --all-tasks --episodes 100
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
from typing import List

import torch

from src.training.evaluator import Evaluator
from src.utils.config import (
    DiffusionPolicyConfig,
    SimulatorConfig,
    TaskConfig,
    load_diffusion_policy_config,
    load_simulator_config,
    load_task_config,
)
from src.vla.diffusion_policy import DiffusionPolicy
from src.vla.text_encoder import CachedTextEmbeddings


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained policy.")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--task", type=str, default=None,
                        help="Task name to evaluate")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Evaluate on all tasks")
    parser.add_argument("--episodes", type=int, default=100,
                        help="Number of episodes per task")
    parser.add_argument("--sim-config", type=str, default="configs/simulator/default.yaml")
    parser.add_argument("--model-config", type=str, default="configs/model/diffusion_policy.yaml")
    parser.add_argument("--text-cache", type=str, default="data/text_embeddings.pt")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save results JSON")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load configs
    sim_config = load_simulator_config(Path(args.sim_config))
    model_config = load_diffusion_policy_config(Path(args.model_config))

    # Load model
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
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint (epoch {ckpt['epoch']})")

    # Load text embeddings
    text_cache = CachedTextEmbeddings(args.text_cache)

    # Determine tasks
    ALL_TASKS = ["embryo", "oocyte", "real_sperm_head", "whole_sperm", "microsphere"]
    if args.all_tasks:
        task_names = ALL_TASKS
    elif args.task:
        task_names = [args.task]
    else:
        parser.error("Specify --task or --all-tasks")

    task_configs: List[TaskConfig] = []
    text_embeddings = {}
    for name in task_names:
        cfg_path = Path(f"configs/simulator/{name}.yaml")
        if cfg_path.exists():
            task_configs.append(load_task_config(cfg_path))
        if name in text_cache:
            text_embeddings[name] = text_cache.get(name)
        else:
            print(f"Warning: no text embedding for '{name}'")

    # Evaluate
    evaluator = Evaluator(sim_config, device=str(device))
    print(f"\nEvaluating on {len(task_configs)} task(s), {args.episodes} episodes each...\n")

    aggregate = evaluator.evaluate_all_tasks(
        policy=model,
        task_configs=task_configs,
        text_embeddings=text_embeddings,
        num_episodes_per_task=args.episodes,
        seed=args.seed,
        obs_horizon=model_config.obs_horizon,
    )

    print(f"\n{'='*50}")
    print(f"OVERALL: success={aggregate.success_rate:.1%}, "
          f"dist={aggregate.mean_final_distance:.2f}+/-{aggregate.std_final_distance:.2f}px, "
          f"smoothness={aggregate.mean_smoothness:.3f}")
    print(f"Episodes: {aggregate.num_episodes}")

    if args.output:
        import json
        import numpy as np

        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, (np.floating, float)):
                    return float(obj)
                if isinstance(obj, np.integer):
                    return int(obj)
                return super().default(obj)

        results = {
            "success_rate": aggregate.success_rate,
            "mean_final_distance": aggregate.mean_final_distance,
            "std_final_distance": aggregate.std_final_distance,
            "mean_smoothness": aggregate.mean_smoothness,
            "per_task": aggregate.per_task,
        }
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2, cls=NpEncoder)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
