
import torch
import torch.nn as nn

# 1 image, 3 channels, 640 × 640
x = torch.randn(1, 3, 640, 640)

# 3 input channels -> 16 output channels
conv = nn.Conv2d(
    in_channels=3,
    out_channels=16,
    kernel_size=3,
    stride=2,
    padding=1
)

y = conv(x)

print("Input shape:", x.shape)
print("Output shape:", y.shape)
print("Parameters:", sum(p.numel() for p in conv.parameters()))
