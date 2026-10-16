"""Tiny-CNN-A: 48x48 grayscale -> neutral/happy/sad logits.

Architecture follows the user's FPGA implementation plan V1.0.
No dataset downloading, training, quantization, or GPU allocation occurs on import.
"""
from __future__ import annotations

import torch
from torch import nn


CLASS_NAMES = ("neutral", "happy", "sad")


class TinyCNN(nn.Module):
    def __init__(self, channels: tuple[int, int, int] = (8, 16, 32), num_classes: int = 3):
        super().__init__()
        if len(channels) != 3 or any(c <= 0 for c in channels):
            raise ValueError("TinyCNN expects three positive channel counts")
        layers: list[nn.Module] = []
        in_ch = 1
        for index, out_ch in enumerate(channels):
            layers.extend((
                nn.Conv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=False),
            ))
            if index < 2:
                layers.append(nn.MaxPool2d(2))
            in_ch = out_ch
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(channels[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[-2:] != (48, 48):
            raise ValueError("Expected tensor shape [N, 1, 48, 48]")
        return self.fc(self.gap(self.features(x)).flatten(1))


def num_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
