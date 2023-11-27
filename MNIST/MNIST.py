import matplotlib.pyplot as plt
import numpy as np
from torchvision.datasets import MNIST
import torch

dataset = MNIST(root='data', train=True, download=True)

for img,label in dataset:
    img = np.array((img))
    print(img.shape, img.dtype)
    break

fig, axes = plt.subplots(2, 5, figsize=(10,5))
for ax_idx, ax in enumerate(axes.flat):
    img, label = dataset[ax_idx]

    ax.imshow(img, cmap='gray')
    ax.set_title(f"Class {label}", fontsize=15)

    ax.axis('off')
    if ax_idx >= 9:break

fig.tight_layout()
plt.show()