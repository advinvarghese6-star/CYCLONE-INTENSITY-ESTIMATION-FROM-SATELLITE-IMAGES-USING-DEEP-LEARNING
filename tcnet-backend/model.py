import torch
import torch.nn as nn

class TCNet(nn.Module):
    """
    TCNet: 4-layer Convolutional Neural Network for Cyclone Intensity Estimation
    Phase 3 Architecture (matches baseline_cnn_v3.pt checkpoint exactly).
    Input: (B, 1, 64, 64) normalized single-channel infrared satellite image
    Output: (B,) predicted maximum sustained wind speed (knots)
    Total parameters: 389,057
    """
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=False),
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(256, 1)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).flatten(1)
        return self.head(x).squeeze(-1)
