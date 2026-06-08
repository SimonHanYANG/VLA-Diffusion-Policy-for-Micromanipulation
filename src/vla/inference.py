"""Closed-loop controller for VLA policy inference."""

from typing import Optional

import numpy as np
import torch

from src.simulator.environment import MicroscopeEnvironment
from src.training.metrics import EpisodeMetrics


class ClosedLoopController:
    """Runs a diffusion policy (or baseline) in closed-loop control.

    Pseudocode:
      obs_history = [env.reset()]
      while not done:
          obs_seq = stack(obs_history[-obs_horizon:])
          action_seq = policy.predict_action(obs_seq, text_emb)
          first_action = action_seq[0, 0]  # (2,)
          obs = env.step(first_action)
          obs_history.append(obs)
          if distance < tolerance: done = True
    """

    def __init__(
        self,
        policy,
        env: MicroscopeEnvironment,
        text_emb: torch.Tensor,
        obs_horizon: int = 2,
        pred_horizon: int = 10,
        device: str = "cpu",
    ):
        self.policy = policy.to(device)
        self.env = env
        self.text_emb = text_emb.to(device)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.device = device

        self.obs_history: list = []
        self.actions: list = []
        self.done = False

    def reset(self, seed: Optional[int] = None) -> np.ndarray:
        """Reset environment and clear history."""
        obs, info = self.env.reset(seed=seed)
        self.obs_history = [obs]
        self.actions = []
        self.done = False
        return obs

    def step(self) -> tuple[np.ndarray, float, bool, dict]:
        """Run one inference step. Returns (obs, reward, done, info)."""
        if self.done:
            return self.obs_history[-1], 0.0, True, {}

        # Build observation sequence
        if len(self.obs_history) < self.obs_horizon:
            pad = self.obs_history[0]
            obs_seq_imgs = [pad] * (self.obs_horizon - len(self.obs_history)) + self.obs_history
        else:
            obs_seq_imgs = self.obs_history[-self.obs_horizon:]

        obs_seq = torch.from_numpy(
            np.stack([img.transpose(2, 0, 1) for img in obs_seq_imgs], axis=0)
        ).float().unsqueeze(0).to(self.device) / 255.0

        text = self.text_emb.unsqueeze(0).to(self.device)

        action_seq = self.policy.predict_action(obs_seq, text)  # (1, H, 2)
        action = action_seq[0, 0].cpu().numpy()  # First action

        obs, reward, done, info = self.env.step(action)
        self.obs_history.append(obs)
        self.actions.append(action)
        self.done = done
        return obs, reward, done, info

    def run_episode(self, seed: Optional[int] = None, max_steps: int = 200) -> EpisodeMetrics:
        """Run a full episode. Returns EpisodeMetrics."""
        self.reset(seed=seed)
        target_label = self.env.target_render.label if self.env.target_render else ""

        for _ in range(max_steps):
            obs, reward, done, info = self.step()
            if done:
                break

        actions_arr = np.array(self.actions) if self.actions else np.zeros((0, 2))
        return EpisodeMetrics(
            success=info.get("distance", float("inf")) < self.env.config.success_tolerance_px,
            final_distance=float(info.get("distance", float("inf"))),
            trajectory_length=len(self.actions),
            mean_step_distance=float(np.mean(np.linalg.norm(actions_arr, axis=1))) if len(self.actions) > 0 else 0.0,
            trajectory_smoothness=EpisodeMetrics.compute_smoothness(actions_arr),
            task_name=target_label,
        )
