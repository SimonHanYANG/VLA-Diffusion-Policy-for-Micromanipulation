from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm

from src.simulator.environment import MicroscopeEnvironment
from src.simulator.targets import create_target_generator, TargetGenerator
from src.training.metrics import EpisodeMetrics, AggregateMetrics, compute_aggregate_metrics
from src.utils.config import SimulatorConfig, TaskConfig


class Evaluator:
    """Batch evaluator for trained policies in simulation."""

    def __init__(
        self,
        sim_config: SimulatorConfig,
        device: str = "cpu",
    ):
        self.sim_config = sim_config
        self.device = device

    @torch.no_grad()
    def evaluate_policy(
        self,
        policy,
        task_config: TaskConfig,
        text_emb: torch.Tensor,
        num_episodes: int = 100,
        seed: int = 0,
        obs_horizon: int = 2,
    ) -> List[EpisodeMetrics]:
        """Evaluate a policy on a single task.

        Args:
            policy: model with predict_action(obs_seq, text_emb) method.
            task_config: task configuration.
            text_emb: (512,) precomputed text embedding for this task.
            num_episodes: number of episodes to evaluate.
            seed: random seed for reproducibility.
            obs_horizon: number of observation frames to stack.

        Returns:
            List of EpisodeMetrics, one per episode.
        """
        target_gen = create_target_generator(
            task_config.target_generator, task_config.target_params
        )
        env = MicroscopeEnvironment(
            config=self.sim_config,
            target_generator=target_gen,
        )

        rng = np.random.default_rng(seed)
        episodes = []

        for ep_idx in tqdm(range(num_episodes), desc=f"Evaluating {task_config.name}", leave=False):
            ep_seed = int(rng.integers(0, 2**31))
            obs, info = env.reset(seed=ep_seed)
            obs_history = [obs]

            all_actions = []
            done = False

            for step in range(self.sim_config.max_steps_per_episode):
                # Build observation sequence
                if len(obs_history) < obs_horizon:
                    # Pad with first observation
                    pad_count = obs_horizon - len(obs_history)
                    obs_seq_imgs = [obs_history[0]] * pad_count + obs_history
                else:
                    obs_seq_imgs = obs_history[-obs_horizon:]

                obs_seq = torch.from_numpy(
                    np.stack([img.transpose(2, 0, 1) for img in obs_seq_imgs], axis=0)
                ).float().unsqueeze(0).to(self.device) / 255.0  # (1, obs_horizon, C, H, W)

                text = text_emb.unsqueeze(0).to(self.device)  # (1, 512)

                # Predict action
                action_seq = policy.predict_action(obs_seq, text)  # (1, pred_horizon, 2)
                action = action_seq[0, 0].cpu().numpy()  # (2,) - first action only

                obs, reward, done, info = env.step(action)
                obs_history.append(obs)
                all_actions.append(action)

                if done:
                    break

            final_dist = float(info.get("distance", float("inf")))
            metrics = EpisodeMetrics(
                success=final_dist < self.sim_config.success_tolerance_px,
                final_distance=final_dist,
                trajectory_length=len(all_actions),
                mean_step_distance=float(np.mean(np.linalg.norm(all_actions, axis=1))) if all_actions else 0.0,
                trajectory_smoothness=EpisodeMetrics.compute_smoothness(np.array(all_actions)) if len(all_actions) > 1 else 0.0,
                task_name=task_config.name,
            )
            episodes.append(metrics)

        return episodes

    def evaluate_all_tasks(
        self,
        policy,
        task_configs: List[TaskConfig],
        text_embeddings: dict,
        num_episodes_per_task: int = 100,
        seed: int = 0,
        obs_horizon: int = 2,
    ) -> AggregateMetrics:
        """Evaluate on all tasks and aggregate results."""
        all_episodes = []
        for tc in task_configs:
            text_emb = text_embeddings.get(tc.name)
            if text_emb is None:
                print(f"Warning: no text embedding for task '{tc.name}', skipping")
                continue
            episodes = self.evaluate_policy(
                policy, tc, text_emb, num_episodes_per_task, seed, obs_horizon
            )
            all_episodes.extend(episodes)

            # Print per-task results
            task_metrics = compute_aggregate_metrics(episodes)
            print(f"  {tc.name}: success={task_metrics.success_rate:.1%}, "
                  f"dist={task_metrics.mean_final_distance:.2f}px, "
                  f"smoothness={task_metrics.mean_smoothness:.3f}")

        return compute_aggregate_metrics(all_episodes)
