"""Action Chunking Transformer (ACT) baseline.

Simplified implementation adapted for 2D microscope action prediction.
Based on the ACT architecture: Transformer encoder-decoder with VAE-style
latent variable for modeling action sequence multimodality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.vla.visual_encoder import VisualEncoder


class ACTPolicy(nn.Module):
    """Action Chunking Transformer baseline.

    Encodes observation into tokens, decodes action sequence with
    a Transformer decoder. Uses a learned query token set for the action chunk.
    """

    def __init__(
        self,
        obs_horizon: int = 2,
        pred_horizon: int = 10,
        action_dim: int = 2,
        visual_backbone: str = "resnet18",
        visual_output_dim: int = 256,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.action_dim = action_dim
        self.d_model = d_model

        # Visual encoder
        self.visual_encoder = VisualEncoder(
            backbone=visual_backbone,
            output_dim=d_model,
            obs_horizon=obs_horizon,
        )

        # Positional embeddings for action queries
        self.action_query = nn.Parameter(torch.randn(1, pred_horizon, d_model) * 0.02)
        self.action_pos_embed = nn.Parameter(torch.randn(1, pred_horizon, d_model) * 0.02)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation="relu", batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.action_head = nn.Linear(d_model, action_dim)

    def forward(self, obs_seq: torch.Tensor) -> torch.Tensor:
        """Predict action sequence.

        Args:
            obs_seq: (B, obs_horizon, C, H, W)

        Returns:
            (B, pred_horizon, action_dim) action sequence.
        """
        B = obs_seq.shape[0]

        # Encode observation
        vis_feat = self.visual_encoder(obs_seq)  # (B, d_model)
        memory = vis_feat.unsqueeze(1)  # (B, 1, d_model) - single memory token

        # Expand action queries
        queries = self.action_query.expand(B, -1, -1)  # (B, pred_horizon, d_model)
        queries = queries + self.action_pos_embed

        # Decode
        output = self.decoder(queries, memory)  # (B, pred_horizon, d_model)
        actions = self.action_head(output)  # (B, pred_horizon, 2)
        return actions

    def compute_loss(self, obs_seq: torch.Tensor, action_seq: torch.Tensor,
                     text_emb: torch.Tensor) -> torch.Tensor:
        """L1 loss on full action sequence."""
        pred = self.forward(obs_seq)
        return F.l1_loss(pred, action_seq)

    @torch.no_grad()
    def predict_action(self, obs_seq: torch.Tensor, text_emb: torch.Tensor) -> torch.Tensor:
        """Predict action sequence.

        Returns:
            (B, pred_horizon, action_dim).
        """
        return self.forward(obs_seq)
