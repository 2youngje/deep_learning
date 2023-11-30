import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_channels, out_channels,kernel_size, padding):
        super(ConvBlock,self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,kernel_size=kernel_size, padding=padding)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x)
        x = self.relu(x)
        return x

class InceptionNaive(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3, ch5x5):
        # input channel 수와 1x1, 3x3, 5x5 branch의 output channel 수를 입력받음
        super(InceptionNaive, self).__init__()
        self.branch1 = ConvBlock(in_channels=in_channels, out_channels=ch1x1,
                                 kernel_size=1, padding=0)
        self.branch2 = ConvBlock(in_channels=in_channels, out_channels=ch3x3,
                                 kernel_size=3, padding=1)
        self.branch3 = ConvBlock(in_channels=in_channels, out_channels=ch5x5,
                                 kernel_size=5, padding=2)
        self.branch4 = nn.MaxPool2d(kernel_size=3, padding=1, stride=1)

    def forward(self, x):
        concat_axis = 1 if x.dim() == 4 else 0
        out_branch1 = self.branch1.forward(x)
        out_branch2 = self.branch2.forward(x)
        out_branch3 = self.branch3.forward(x)
        out_branch4 = self.branch4.forward(x)
        to_concat = (out_branch1, out_branch2, out_branch3, out_branch4)
        x = torch.cat(tensors=to_concat, dim=concat_axis)
        return x


    def forward(self,x):
        out_branch1 = self.branch1(x)
        out_branch2 = self.branch2(x)
        out_branch3 = self.branch3(x)
        out_branch4 = self.branch4(x)
        x = torch.concat([out_branch1,out_branch2,out_branch3,out_branch4], dim=0)
        return x

input_tensor = torch.randn(192,100,100)

branch1 = InceptionNaive(in_channels=192, ch1x1=64, ch3x3=128, ch5x5=32)

output = branch1(input_tensor)

print(output.shape)

# branch1 = ConvBlock(192,64)
#
# out_branch1 = branch1.forward(data)
#
# print(out_branch1.shape)
#
# branch2 = ConvBlock(192,128)
#
# out_branch2 = branch2.forward(data)
#
# print(out_branch2.shape)
#
# branch3 = ConvBlock(192,32)
#
# out_branch3 = branch3.forward(data)
#
# print(out_branch3.shape)
#
# branch4 = ConvBlock(192,192)
#
# out_branch4 = branch4.forward(data)
#
# out_inception = torch.concat([out_branch1,out_branch2,out_branch3,out_branch4])
#
# print(out_inception.shape)

