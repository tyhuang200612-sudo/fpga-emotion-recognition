"""Tiny-CNN-C: hardware-oriented 4-layer INT8-ready FP32 training reference.

48x48 grayscale -> conv16,pool -> conv32,pool -> conv64 -> conv64
-> global average pool -> neutral/happy/sad logits.

All kernels are ordinary 3x3 conv+BN+ReLU; no desktop-only attention, giant
backbone, or transformer. BatchNorm must be folded before FPGA INT8 export.
"""
from __future__ import annotations
import torch
from torch import nn

CLASS_NAMES_C = ("neutral", "happy", "sad")
C_CHANNELS = (16, 32, 64, 64)


class TinyCNNC(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        in_channels = 1
        for layer, channels in enumerate(C_CHANNELS):
            layers.extend((
                nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=False),
            ))
            if layer < 2:
                layers.append(nn.MaxPool2d(2))
            in_channels = channels
        self.features = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, len(CLASS_NAMES_C))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (1, 48, 48):
            raise ValueError("TinyCNNC expects [N,1,48,48]")
        return self.fc(self.gap(self.features(x)).flatten(1))


def resource_estimates():
    channels = C_CHANNELS
    hws = (48, 24, 12, 12)
    in_cs = (1, 16, 32, 64)
    conv_weights = sum(i * o * 9 for i, o in zip(in_cs, channels))
    weights = conv_weights + 64 * 3
    mac = sum(hw * hw * i * o * 9 for hw, i, o in zip(hws, in_cs, channels)) + 64 * 3
    return {"weight_count": weights, "int8_weight_bytes": weights,
            "macs_per_frame": mac, "input_shape": [1, 48, 48],
            "bn_fusion_required": True, "int8_activation_quantization_required": True}
