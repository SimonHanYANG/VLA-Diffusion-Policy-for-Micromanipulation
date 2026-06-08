import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vla.visual_encoder import VisualEncoder


class BCPolicy(nn.Module):
    """Behavior Cloning baseline: CNN (ResNet-18) -> MLP -> single-step (dx, dy) prediction.

    Trained with MSE loss. No temporal modeling beyond observation stacking.
    """

    def __init__(
        self,
        obs_horizon: int = 2,
        action_dim: int = 2,
        visual_backbone: str = "resnet18",
        visual_output_dim: int = 256,
        hidden_dims: list | None = None,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        self.obs_horizon = obs_horizon
        self.action_dim = action_dim

        self.visual_encoder = VisualEncoder(
            backbone=visual_backbone,
            output_dim=visual_output_dim,
            obs_horizon=obs_horizon,
        )

        layers = []
        in_dim = visual_output_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, action_dim))

        self.mlp = nn.Sequential(*layers)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Predict single action from observation.

        Args:
            obs_seq: (B, obs_horizon, C, H, W)

        Returns:
            (B, 2) predicted (dx, dy).
        """
        feat = self.visual_encoder(obs_seq)  # (B, visual_output_dim)
        action = self.mlp(feat)  # (B, 2)
        return action

    def compute_loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
                     text_emb: torch.Tensor) -> torch.Tensor:
        """MSE loss on the first action step. Ignores text_emb."""
        pred = self.forward(obs_seq)
        target = action_seq[:, 0, :]
        return F.mse_loss(pred, target)

    @torch.no_grad()
    def predict_action(self, obs_seq: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Predict action sequence (single step repeated for compatibility).

        Returns:
            (B, 1, 2) — BC predicts only one step.
        """
        pred = self.forward(obs_seq)
        return pred.unsqueeze(1)
