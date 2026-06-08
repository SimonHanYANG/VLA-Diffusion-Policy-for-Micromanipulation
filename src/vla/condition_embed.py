import torch
import torch.nn as nn


class ConditionalEmbedding(nn.Module):
    """Fuses visual features and text features into a condition vector.

    Architecture:
      Concat(visual, text) -> Linear -> ReLU -> Dropout
                           -> Linear -> ReLU -> Dropout
                           -> Linear(output_dim) -> LayerNorm
    """

    def __init__(
        self,
        visual_dim: int = 256,
        text_dim: int = 512,
        hidden_dims: list[int] | None = None,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [512, 256]

        input_dim = visual_dim + text_dim
        layers = []
        in_dim = input_dim

        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim

        layers.append(nn.Linear(in_dim, output_dim))
        layers.append(nn.LayerNorm(output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, visual_feat: torch.Tensor, text_feat: torch.Tensor) -> torch.Tensor:
        """Fuse visual and text features.

        Args:
            visual_feat: (B, visual_dim) visual features.
            text_feat: (B, text_dim) text embeddings.

        Returns:
            (B, output_dim) condition vector.
        """
        fused = torch.cat([visual_feat, text_feat], dim=-1)
        return self.net(fused)
