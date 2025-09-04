import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class SEBlock2D(nn.Module):
    """
    2D Squeeze-and-Excitation block for adaptive channel weighting.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, reduced, bias=True)
        self.fc2 = nn.Linear(reduced, channels, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        b, c, h, w = x.size()
        squeeze = x.view(b, c, -1).mean(dim=2)  # Global avg over H,W
        excitation = F.relu(self.fc1(squeeze), inplace=True)
        excitation = torch.sigmoid(self.fc2(excitation))
        excitation = excitation.view(b, c, 1, 1)
        return x * excitation


class BloodFlowCNN2D(nn.Module):
    """
    Deep 2D CNN for speckle pattern blood flow estimation.
    Input: (B, 1, H, W)
    Output: (B,) predicted flow rates.
    """
    def __init__(
        self,
        input_channels: int = 1,
        base_channels: int = 32,
        reduction: int = 16,
        dropout_rate: float = 0.4,
        input_norm: bool = True,
    ):
        super().__init__()

        self.input_norm = nn.BatchNorm2d(input_channels) if input_norm else nn.Identity()

        # 2D CNN encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.LeakyReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.LeakyReLU(inplace=True),

            SEBlock2D(base_channels * 4, reduction=reduction),

            nn.Conv2d(base_channels * 4, base_channels * 8, kernel_size=3, padding=1),
            nn.BatchNorm2d(base_channels * 8),
            nn.LeakyReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        # Fully connected regression head
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(base_channels * 8, base_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(base_channels * 4, 1)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_input_shape(x)
        x = self.input_norm(x)
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x.squeeze(-1)

    def _initialize_weights(self):
        """Kaiming-normal initialization for Conv and Linear layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    @staticmethod
    def _validate_input_shape(x: torch.Tensor):
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {x.shape}")
        if x.size(1) != 1:
            raise ValueError(f"Expected single channel input, got {x.size(1)} channels.")

    def freeze_encoder(self):
        """Freeze encoder layers for transfer learning."""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze encoder layers."""
        for param in self.encoder.parameters():
            param.requires_grad = True


def get_model(**kwargs) -> BloodFlowCNN2D:
    """Factory function to create 2D BloodFlowCNN."""
    return BloodFlowCNN2D(**kwargs)


if __name__ == "__main__":
    # Quick sanity check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model().to(device)
    dummy_input = torch.randn(2, 1, 128, 128).to(device)
    with torch.no_grad():
        output = model(dummy_input)
    print("Output shape:", output.shape)
