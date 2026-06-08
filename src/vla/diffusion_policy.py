import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vla.condition_embed import ConditionalEmbedding
from src.vla.diffusion_unet import DiffusionUNet1D
from src.vla.noise_scheduler import NoiseScheduler
from src.vla.visual_encoder import VisualEncoder


class DiffusionPolicy(nn.Module):
    """Full conditional diffusion policy for action prediction.

    Training:
      1. Encode observation -> visual features
      2. Fuse visual + text -> condition vector
      3. Sample noise eps ~ N(0,I), timestep t ~ U(0,T)
      4. Noisy actions = scheduler.add_noise(actions, eps, t)
      5. noise_pred = unet(noisy_actions, t, cond)
      6. loss = MSE(noise_pred, eps)

    Inference (DDIM, closed-loop):
      1. Encode observation -> condition
      2. Start from x_T ~ N(0,I)
      3. Iteratively denoise through DDIM steps
      4. Return x_0 as predicted action sequence
    """

    def __init__(
        self,
        obs_horizon: int = 2,
        pred_horizon: int = 10,
        action_dim: int = 2,
        visual_backbone: str = "resnet18",
        visual_output_dim: int = 256,
        text_dim: int = 512,
        condition_hidden_dims: list | None = None,
        condition_output_dim: int = 256,
        unet_dims: list | None = None,
        num_diffusion_steps: int = 100,
        num_ddim_steps: int = 10,
        beta_schedule: str = "cosine",
    ):
        super().__init__()

        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.num_ddim_steps = num_ddim_steps

        # Modules
        self.visual_encoder = VisualEncoder(
            backbone=visual_backbone,
            output_dim=visual_output_dim,
            obs_horizon=obs_horizon,
        )
        self.condition_embed = ConditionalEmbedding(
            visual_dim=visual_output_dim,
            text_dim=text_dim,
            hidden_dims=condition_hidden_dims,
            output_dim=condition_output_dim,
        )
        self.noise_scheduler = NoiseScheduler(
            num_train_steps=num_diffusion_steps,
            num_ddim_steps=num_ddim_steps,
            schedule=beta_schedule,
        )
        self.diffusion_unet = DiffusionUNet1D(
            action_dim=action_dim,
            horizon=pred_horizon,
            cond_dim=condition_output_dim,
            dims=unet_dims or [64, 128, 256],
        )

    def compute_loss(
        self,
        obs_seq: torch.Tensor,
        action_seq: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """Training step. Returns scalar MSE loss.

        Args:
            obs_seq: (B, obs_horizon, C, H, W) image observations.
            action_seq: (B, pred_horizon, action_dim) ground truth actions.
            text_emb: (B, text_dim) precomputed CLIP text embeddings.

        Returns:
            scalar loss.
        """
        B = obs_seq.shape[0]
        device = obs_seq.device

        # Encode
        visual_feat = self.visual_encoder(obs_seq)
        cond = self.condition_embed(visual_feat, text_emb)

        # Sample noise and timestep
        noise = torch.randn(B, self.action_dim, self.pred_horizon, device=device)
        t = torch.randint(0, self.noise_scheduler.num_train_steps, (B,), device=device)

        # Forward diffusion
        actions = action_seq.permute(0, 2, 1)  # (B, action_dim, horizon)
        noisy_actions = self.noise_scheduler.add_noise(actions, noise, t)

        # Predict noise
        noise_pred = self.diffusion_unet(noisy_actions, t, cond)

        loss = F.mse_loss(noise_pred, noise)
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        obs_seq: torch.Tensor,
        text_emb: torch.Tensor,
    ) -> torch.Tensor:
        """DDIM inference. Returns predicted action sequence.

        Args:
            obs_seq: (B, obs_horizon, C, H, W) image observations.
            text_emb: (B, text_dim) text embeddings.

        Returns:
            (B, pred_horizon, action_dim) predicted action sequence.
        """
        B = obs_seq.shape[0]
        device = obs_seq.device

        # Encode
        visual_feat = self.visual_encoder(obs_seq)
        cond = self.condition_embed(visual_feat, text_emb)

        # DDIM sampling
        timesteps = self.noise_scheduler.get_ddim_timesteps(device)
        x_t = torch.randn(B, self.action_dim, self.pred_horizon, device=device)

        for i in range(len(timesteps) - 1):
            t = timesteps[i].item()
            t_prev = timesteps[i + 1].item()

            # Predict noise
            t_batch = torch.full((B,), t, device=device, dtype=torch.long)
            noise_pred = self.diffusion_unet(x_t, t_batch, cond)

            # DDIM step
            x_t = self.noise_scheduler.ddim_step(x_t, noise_pred, t, t_prev)

        # x_t is now x_0: (B, action_dim, horizon)
        action_seq = x_t.permute(0, 2, 1)  # (B, horizon, action_dim)
        return action_seq
