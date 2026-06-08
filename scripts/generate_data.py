"""Generate expert demonstration trajectories for all tasks.

Usage:
  python scripts/generate_data.py --task microsphere --num-trajectories 100
  python scripts/generate_data.py --all-tasks --num-trajectories 5000 --workers 4
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import multiprocessing as mp
from typing import List

from src.simulator.background import PerlinNoiseBackground
from src.simulator.expert_generator import ExpertDemonstrationGenerator
from src.simulator.targets import create_target_generator
from src.utils.config import (
    SimulatorConfig,
    TaskConfig,
    load_simulator_config,
    load_task_config,
)


def generate_task(
    task_name: str,
    sim_config: SimulatorConfig,
    task_config: TaskConfig,
    output_root: Path,
    num_trajectories: int,
    seed: int,
) -> Path:
    """Generate trajectories for a single task."""
    target_gen = create_target_generator(
        task_config.target_generator, task_config.target_params
    )
    gen = ExpertDemonstrationGenerator(
        config=sim_config,
        target_generator=target_gen,
        background=PerlinNoiseBackground(),
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
                        help="Task name (microsphere, yeast, sperm_head, sperm_tail)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="Generate data for all tasks")
    parser.add_argument("--num-trajectories", type=int, default=5000,
                        help="Number of trajectories per task")
    parser.add_argument("--workers", type=int, default=1,
                        help="Number of parallel workers")
    parser.add_argument("--output-root", type=str, default="data/trajectories",
                        help="Output root directory")
    parser.add_argument("--sim-config", type=str, default="configs/simulator/default.yaml",
                        help="Path to simulator config YAML")
    parser.add_argument("--seed", type=int, default=0,
                        help="Base random seed")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    sim_config = load_simulator_config(Path(args.sim_config))

    if args.task:
        task_names = [args.task]
    elif args.all_tasks:
        task_names = ["microsphere", "yeast", "sperm_head", "sperm_tail"]
    else:
        parser.error("Specify --task or --all-tasks")

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
                    (tc.name, sim_config, tc, output_root, args.num_trajectories, args.seed + i * 10000),
                )
                results.append(r)
            for r in results:
                r.get()
    else:
        for i, tc in enumerate(task_configs):
            print(f"\n{'='*50}")
            print(f"Task: {tc.name} | Instruction: {tc.instruction}")
            print(f"{'='*50}")
            out_dir = generate_task(
                tc.name, sim_config, tc, output_root,
                args.num_trajectories, args.seed + i * 10000,
            )
            print(f"Saved to: {out_dir}")

    print("\nDone! All trajectories generated.")


if __name__ == "__main__":
    main()
