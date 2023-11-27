import torch
import torch.nn as nn

input_tensor = torch.randn(size=(1, 1, 32, 32))

# kernel_size를 (5, 5)로 변경
conv = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(5, 5), padding=2, stride=1)
output_tensor = conv(input_tensor)

print(output_tensor.shape)