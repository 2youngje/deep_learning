import torch.nn as nn
from torchsummary import summary

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_layers):
        super(ConvBlock, self).__init__()


        #처음 conv layer는 in_channels를 사용
        self.layers = [
            nn.Conv2d(in_channels = in_channels, out_channels= out_channels,
                      kernel_size=3,padding=1),
            nn.ReLU()
        ]

        #n_layers가 2 이상일 때 동작하는 코드
        for _ in range(n_layers - 1):
            self.layers.append(nn.Conv2d(in_channels= out_channels, out_channels=out_channels,
                                         kernel_size=3, padding=1))
            self.layers.append(nn.ReLU())
            in_channels = out_channels

        # 마지막에 max pooling 추가
        self.layers.append(nn.MaxPool2d(kernel_size=2, stride=2))

        # list에 들어있는 layer를 풀어 nn.Sequential에 입력
        self.layers = nn.Sequential(*self.layers)

    def forward(self,x):
         x = self.layers(x)
         return x

block = ConvBlock(3,64,4)
summary(block, input_size=(3,100,100))