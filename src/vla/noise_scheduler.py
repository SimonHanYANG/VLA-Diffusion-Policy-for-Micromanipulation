import numpy as np
import torch
import torch.nn as nn


class NoiseScheduler(nn.Module):
    """Manages the beta schedule for DDPM/DDIM diffusion processes.

    Precomputes all alpha/beta/alpha_bar values at init.
    Supports cosine and linear schedules.
    """

    def __init__(
        self,
        num_train_steps: int = 100,
        num_ddim_steps: int = 10,
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        schedule: str = "cosine",
    ):
        super().__init__()
        self.num_train_steps = num_train_steps
        self.num_ddim_steps = num_ddim_steps

        if schedule == "cosine":
            betas = self._cosine_beta_schedule(num_train_steps)
        elif schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_steps)
        else:
            raise ValueError(f"Unknown beta schedule: '{schedule}'. Use 'cosine' or 'linear'.")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps + 1
        t = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos((t / timesteps + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.02)

    def add_noise(
        self, x_0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """Forward diffusion: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise.

        Args:
            x_0: (B, action_dim, horizon) clean actions.
            noise: (B, action_dim, horizon) random Gaussian noise.
            t: (B,) timestep indices.

        Returns:
            (B, action_dim, horizon) noisy actions.
        """
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]  # (B,)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]  # (B,)

        # Reshape for broadcasting: (B,) -> (B, 1, 1)
        sqrt_alpha_bar = sqrt_alpha_bar[:, None, None]
        sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar[:, None, None]

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

    def get_ddim_timesteps(self, device: torch.device) -> torch.Tensor:
        """Return the DDIM sampling timesteps in reverse order."""
        if self.num_ddim_steps <= 0:
            raise ValueError(f"num_ddim_steps must be positive, got {self.num_ddim_steps}")
        if self.num_ddim_steps > self.num_train_steps:
            raise ValueError(
                f"num_ddim_steps ({self.num_ddim_steps}) cannot exceed "
                f"num_train_steps ({self.num_train_steps})"
            )
        step_size = self.num_train_steps // self.num_ddim_steps
        timesteps = torch.arange(0, self.num_train_steps, step_size, device=device)
        if timesteps[-1] != self.num_train_steps - 1:
            timesteps = torch.cat([timesteps, torch.tensor([self.num_train_steps - 1], device=device)])
        return timesteps.flip(0)

    @torch.no_grad()
    def ddim_step(
        self,
        x_t: torch.Tensor,
        noise_pred: torch.Tensor,
        t: int,
        t_prev: int,
    ) -> torch.Tensor:
        """Single DDIM reverse diffusion step.

        Args:
            x_t: (B, action_dim, horizon) current noisy sample.
            noise_pred: (B, action_dim, horizon) predicted noise.
            t: current timestep index.
            t_prev: previous (larger) timestep index for next step.

        Returns:
            (B, action_dim, horizon) denoised sample at t_prev.
        """
        alpha_bar_t = self.alphas_cumprod[t]
        alpha_bar_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)

        # Predict x_0
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

        # Direction pointing to x_t
        dir_xt = torch.sqrt(1.0 - alpha_bar_t_prev) * noise_pred

        # x_{t-1} = sqrt(alpha_bar_{t-1}) * x_0_pred + sqrt(1-alpha_bar_{t-1}) * noise_pred
        x_prev = torch.sqrt(alpha_bar_t_prev) * x_0_pred + dir_xt

        return x_prev
