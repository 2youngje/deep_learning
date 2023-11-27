import torch
import torch.nn as nn
from torchvision.datasets import MNIST

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2, stride=1)
        self.activation1 = nn.Tanh()
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.activation2 = nn.Tanh()
        self.avg_pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.activation3 = nn.Tanh()

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.activation4 = nn.Tanh()

        self.fc2 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation1(x)
        x = self.avg_pool1(x)

        x = self.conv2(x)
        x = self.activation2(x)
        x = self.avg_pool2(x)

        x = self.conv3(x)
        x = self.activation3(x)

        x = x.reshape(x.shape[0], -1)

        x = self.fc1(x)
        x = self.activation4(x)

        x = self.fc2(x)

        return x

dataset = MNIST(root='data', train=True, download=True)

input_tensor = torch.randn(size=(10, 1, 28, 28))

CNN = Model()
a = CNN.forward(dataset)

print(a.shape)