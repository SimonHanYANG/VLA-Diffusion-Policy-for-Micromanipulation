"""Generate expert demonstration trajectories using real cell images.

Usage:
  # Generate all tasks (embryo, oocyte, real_sperm_head, whole_sperm, microsphere)
  python scripts/generate_data.py --num-trajectories 100

  # Generate a specific task
  python scripts/generate_data.py --task embryo --num-trajectories 50

  # Generate with multiple workers
  python scripts/generate_data.py --num-trajectories 500 --workers 4
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import multiprocessing as mp
from typing import List

from src.simulator.background import BackgroundGeneratorFactory
from src.simulator.expert_generator import ExpertDemonstrationGenerator
from src.simulator.targets import create_target_generator
from src.utils.config import (
    SimulatorConfig,
    TaskConfig,
    load_simulator_config,
    load_task_config,
)

# All available tasks (real cell images + procedural microsphere)
ALL_TASKS = ["embryo", "oocyte", "real_sperm_head", "whole_sperm", "microsphere"]


def generate_task(
    task_name: str,
    sim_config: SimulatorConfig,
    task_config: TaskConfig,
    output_root: Path,
    num_trajectories: int,
    seed: int,
    background_type: str = "image",
    background_dir: str | None = None,
) -> Path:
    """Generate trajectories for a single task."""
    target_gen = create_target_generator(
        task_config.target_generator, task_config.target_params
    )

    # Create background generator
    bg_image_dir = Path(background_dir) if background_dir else None
    background = BackgroundGeneratorFactory.create(
        background_type=background_type, image_dir=bg_image_dir
    )

    gen = ExpertDemonstrationGenerator(
        config=sim_config,
        target_generator=target_gen,
        background=background,
        pid_kp=task_config.pid_kp,
        pid_ki=task_config.pid_ki,
        pid_kd=task_config.pid_kd,
    )
    output_dir = output_root / task_name
    gen.generate_trajectories(
        num_trajectories=num_trajectories,
        output_dir=output_dir,
        task_name=task_name,
        seed=seed,
    )
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Generate expert demonstration trajectories.")
    parser.add_argument("--task", type=str, default=None,
                        help=f"Task name. Available: {ALL_TASKS}")
    parser.add_argument("--all-tasks", action="store_true", default=True,
                        help="Generate data for all tasks (default)")
    parser.add_argument("--num-trajectories", type=int, default=100,
                        help="Number of trajectories per task")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--output-root", type=str, default="data/trajectories",
                        help="Output root directory")
    parser.add_argument("--sim-config", type=str, default="configs/simulator/clean_640.yaml",
                        help="Path to simulator config YAML")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--background-type", type=str, default="image",
                        choices=["perlin", "image"],
                        help="Background type: 'image' for real microscopy backgrounds (default)")
    parser.add_argument("--background-dir", type=str, default="data/backgrounds",
                        help="Directory containing background images")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sim_config = load_simulator_config(Path(args.sim_config))

    if args.task:
        task_names = [args.task]
    else:
        task_names = ALL_TASKS

    # Load task configs
    task_configs: List[TaskConfig] = []
    for name in task_names:
        cfg_path = Path(f"configs/simulator/{name}.yaml")
        if cfg_path.exists():
            task_configs.append(load_task_config(cfg_path))
        else:
            print(f"Warning: config not found for '{name}', skipping")

    if not task_configs:
        print("No tasks to generate.")
        return

    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            results = []
            for i, tc in enumerate(task_configs):
                r = pool.apply_async(
                    generate_task,
                    (tc.name, sim_config, tc, output_root, args.num_trajectories, args.seed + i * 10000,
                     args.background_type, args.background_dir),
                )
                results.append(r)
            for r in results:
                r.get()
    else:
        for i, tc in enumerate(task_configs):
            print(f"\n{'='*50}")
            print(f"Task: {tc.name} | Instruction: {tc.instruction}")
            print(f"Background: {args.background_type}")
            print(f"{'='*50}")
            out_dir = generate_task(
                tc.name, sim_config, tc, output_root,
                args.num_trajectories, args.seed + i * 10000,
                args.background_type, args.background_dir,
            )
            print(f"Saved to: {out_dir}")

    print("\nDone! All trajectories generated.")


if __name__ == "__main__":
    main()
