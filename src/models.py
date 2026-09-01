"""
Model definitions for the Face Blurring project.

- BackboneNN: VGG-style convolutional backbone ending at 512 channels.
- ClassificationNN: Global-pool + 2-layer MLP for 3-way classification.
- RegressionNN: Conv reduction + FC block for 4-parameter bbox regression.
"""

import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    Basic conv block: Conv-BN-ReLU -> Conv-BN-ReLU -> MaxPool.

    Input:  [B, in_channels, H, W]
    Output: [B, out_channels, H/2, W/2]
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.convblock(x)


class BackboneNN(nn.Module):
    """
    VGG-style backbone for face localization.

    Input:  [B, 3, 224, 224]
    Output: [B, 512, 4, 4]

    Architecture:
        5 x ConvBlock (3->16->32->64->128->256) with 2x downsampling each
        Final strided 3x3 conv: 256 -> 512, stride=2
    """

    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvBlock(in_channels, 16),   # 224 -> 112
            ConvBlock(16, 32),            # 112 -> 56
            ConvBlock(32, 64),            # 56  -> 28
            ConvBlock(64, 128),           # 28  -> 14
            ConvBlock(128, 256),          # 14  -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=2, bias=True),  # 7 -> 4
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ClassificationNN(nn.Module):
    """
    Classification head: global average pool + 2-layer MLP.

    Input:  [B, 512, 4, 4]
    Output: [B, num_classes] (logits)
    """

    def __init__(self, in_features: int = 512, num_classes: int = 3):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten(start_dim=1)
        self.fc1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


class RegressionNN(nn.Module):
    """
    Regression head for bounding-box coordinates.

    Input:  [B, 512, 4, 4]
    Output: [B, 4] normalized coordinates in [0, 1] (x1, y1, w, h)

    Architecture:
        - Conv reduction: 512 -> 256 -> 128 (3x3 convs, BN, ReLU)
        - Flatten: [B, 128, 4, 4] -> [B, 128*4*4]
        - FC block: 2048 -> 512 -> 256 -> 128 -> 4
        - Sigmoid activation to constrain outputs to [0, 1]
    """

    def __init__(self, in_channels: int = 512):
        super().__init__()
        self.conv_reduce = nn.Sequential(
            nn.Conv2d(in_channels, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.fc_block = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_reduce(x)
        x = self.flatten(x)
        x = self.fc_block(x)
        return torch.sigmoid(x)