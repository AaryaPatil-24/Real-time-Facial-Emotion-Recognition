"""CNN models for 48x48 grayscale facial emotion recognition.

``EmotionCNN`` is the original compact VGG-style baseline (1.77M parameters).
``EmotionResNet`` is a wider residual model with squeeze-excitation blocks that
trains to a higher accuracy while still running comfortably on CPU.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def conv_block(cin: int, cout: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(cin, cout, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
    )


class SqueezeExcitation(nn.Module):
    """Cheap channel attention: re-weight feature maps by global context."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(4, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = x.mean(dim=(2, 3))
        scale = torch.relu(self.fc1(scale))
        scale = torch.sigmoid(self.fc2(scale))
        return x * scale[:, :, None, None]


class ResidualBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int = 1, se: bool = True) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.se = SqueezeExcitation(cout) if se else nn.Identity()
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or cin != cout:
            self.shortcut = nn.Sequential(
                nn.Conv2d(cin, cout, 1, stride, bias=False),
                nn.BatchNorm2d(cout),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return torch.relu(out + self.shortcut(x))


class EmotionCNN(nn.Module):
    """Input [B, 1, 48, 48] in [-1, 1]; output 7 logits.

    ``features[-3]`` is the last 3x3 convolution, used as the Grad-CAM target.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.4, width: int = 32) -> None:
        super().__init__()
        w = width
        self.features = nn.Sequential(
            conv_block(1, w),
            conv_block(w, w),
            nn.MaxPool2d(2),          # 48 -> 24
            nn.Dropout(0.25),
            conv_block(w, w * 2),
            conv_block(w * 2, w * 2),
            nn.MaxPool2d(2),          # 24 -> 12
            nn.Dropout(0.25),
            conv_block(w * 2, w * 4),
            conv_block(w * 4, w * 4),
            nn.MaxPool2d(2),          # 12 -> 6
            nn.Dropout(0.30),
            conv_block(w * 4, w * 8),  # index -3, Grad-CAM target
            nn.MaxPool2d(2),           # 6 -> 3
            nn.Dropout(0.30),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(w * 8 * 3 * 3, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes),
        )
        self._init_weights()

    @property
    def cam_target(self) -> nn.Module:
        return self.features[-3]

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class EmotionResNet(nn.Module):
    """Input [B, 1, 48, 48] in [-1, 1]; output 7 logits.

    A compact residual network: two stem convolutions at full resolution, then
    three stages at 24x24, 12x12 and 6x6 with squeeze-excitation attention.
    Global average pooling keeps the classifier small and regularised.
    """

    def __init__(self, num_classes: int = 7, dropout: float = 0.35, width: int = 32) -> None:
        super().__init__()
        w = width
        self.stem = nn.Sequential(
            conv_block(1, w),
            conv_block(w, w),
        )
        self.stage1 = self._stage(w, w * 2, stride=2)
        self.stage2 = self._stage(w * 2, w * 4, stride=2)
        self.stage3 = self._stage(w * 4, w * 8, stride=2)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(w * 8, num_classes),
        )
        self._init_weights()

    @staticmethod
    def _stage(cin: int, cout: int, stride: int) -> nn.Sequential:
        return nn.Sequential(
            ResidualBlock(cin, cout, stride=stride),
            ResidualBlock(cout, cout),
        )

    @property
    def cam_target(self) -> nn.Module:
        return self.stage3[-1].conv2

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)


ARCHITECTURES = {"cnn": EmotionCNN, "resnet": EmotionResNet}


def build_model(
    arch: str = "cnn",
    num_classes: int = 7,
    dropout: float = 0.4,
    width: int = 32,
) -> nn.Module:
    key = arch.lower().replace("_", "")
    key = {"emotioncnn": "cnn", "emotionresnet": "resnet"}.get(key, key)
    if key not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture {arch!r}; choose from {sorted(ARCHITECTURES)}")
    return ARCHITECTURES[key](num_classes=num_classes, dropout=dropout, width=width)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
