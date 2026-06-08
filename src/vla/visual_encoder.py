import torch
import torch.nn as nn
import torchvision.models as models


class VisualEncoder(nn.Module):
    """ResNet-18 backbone -> 256-dim visual feature vector.

    Observation history (obs_horizon frames) is handled by stacking frames
    along the channel dimension (e.g., 2 RGB frames = 6 channels) and passing
    through a modified first convolution layer.
    """

    def __init__(
        self,
        backbone: str = "resnet18",
        output_dim: int = 256,
        obs_horizon: int = 2,
        pretrained: bool = True,
    ):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.output_dim = output_dim

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            self.backbone = models.resnet18(weights=weights)
            feature_dim = 512

            # Modify first conv to accept obs_horizon * 3 channels
            in_channels = obs_horizon * 3
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            # Initialize new channels by replicating pretrained weights
            if pretrained:
                with torch.no_grad():
                    for i in range(in_channels):
                        src_channel = i % 3
                        self.backbone.conv1.weight[:, i] = old_conv.weight[:, src_channel].clone()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.backbone.fc = nn.Identity()
        self.backbone.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.projection = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Encode observation sequence into a single feature vector.

        Args:
            images: (B, obs_horizon, C, H, W) tensor.

        Returns:
            (B, output_dim) feature vector.
        """
        B, T, C, H, W = images.shape
        # Stack frames along channel dim: (B, T*C, H, W)
        x = images.reshape(B, T * C, H, W)

        features = self.backbone(x)  # (B, 512)
        features = self.projection(features)  # (B, output_dim)
        return features
