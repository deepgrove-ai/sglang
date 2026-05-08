"""
Total FLOPs for forward and backward pass using fvcore.
Backward pass FLOPs are approximated as 2x the forward pass.
"""

import torch
import torch.nn as nn
from fvcore.nn import FlopCountAnalysis


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(64, 10)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


model = SmallCNN()
inputs = (torch.randn(1, 3, 32, 32),)

flops = FlopCountAnalysis(model, inputs)
flops.unsupported_ops_warnings(False)
flops.uncalled_modules_warnings(False)

forward_flops = flops.total()
backward_flops = 2 * forward_flops  # standard approximation

print(f"Forward  FLOPs: {forward_flops:,}")
print(f"Backward FLOPs: {backward_flops:,}")