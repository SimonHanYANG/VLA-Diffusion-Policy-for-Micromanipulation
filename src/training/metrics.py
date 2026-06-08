from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class EpisodeMetrics:
    success: bool = False
    final_distance: float = 0.0
    trajectory_length: int = 0
    mean_step_distance: float = 0.0
    trajectory_smoothness: float = 0.0
    task_name: str = ""

    @staticmethod
    def compute_smoothness(actions: np.ndarray) -> float:
        """Mean absolute angular change between consecutive action vectors.
        Lower = smoother. Range [0, pi].
        """
        if len(actions) < 2:
            return 0.0
        diffs = np.diff(actions, axis=0)  # (T-1, 2)
        norm1 = np.linalg.norm(actions[:-1], axis=1) + 1e-8
        norm2 = np.linalg.norm(actions[1:], axis=1) + 1e-8
        cos_angles = np.sum(actions[:-1] * actions[1:], axis=1) / (norm1 * norm2)
        cos_angles = np.clip(cos_angles, -1.0, 1.0)
        angles = np.arccos(cos_angles)
        return float(np.mean(np.abs(angles)))


@dataclass
class AggregateMetrics:
    success_rate: float = 0.0
    mean_final_distance: float = 0.0
    std_final_distance: float = 0.0
    mean_trajectory_length: float = 0.0
    mean_smoothness: float = 0.0
    num_episodes: int = 0
    per_task: dict = field(default_factory=dict)


def compute_aggregate_metrics(episodes: List[EpisodeMetrics]) -> AggregateMetrics:
    """Compute aggregate statistics from a list of episode metrics."""
    if not episodes:
        return AggregateMetrics(num_episodes=0)

    n = len(episodes)
    successes = sum(1 for e in episodes if e.success)
    distances = [e.final_distance for e in episodes]
    lengths = [e.trajectory_length for e in episodes]
    smoothness = [e.trajectory_smoothness for e in episodes]

    # Per-task breakdown
    per_task = {}
    task_names = set(e.task_name for e in episodes)
    for task in task_names:
        task_eps = [e for e in episodes if e.task_name == task]
        task_success = sum(1 for e in task_eps if e.success) / len(task_eps)
        task_dist = np.mean([e.final_distance for e in task_eps])
        per_task[task] = {"success_rate": task_success, "mean_distance": float(task_dist)}

    return AggregateMetrics(
        success_rate=successes / n,
        mean_final_distance=float(np.mean(distances)),
        std_final_distance=float(np.std(distances)),
        mean_trajectory_length=float(np.mean(lengths)),
        mean_smoothness=float(np.mean(smoothness)),
        num_episodes=n,
        per_task=per_task,
    )
