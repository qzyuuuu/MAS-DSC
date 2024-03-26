import torch
from torch import nn
from conv import Conv2dSamePad, ConvTranspose2dSamePad
import sys
sys.path.append('.')

def conv_relu(in_channel, out_channel, kernel, stride=2):
    layer = nn.Sequential(
        Conv2dSamePad(kernel, stride),
        nn.Conv2d(in_channel, out_channel, kernel, stride),
        nn.ReLU(True)
    )
    return layer

def transpose_conv_relu(in_channel, out_channel, kernel, stride=2):
    layer = nn.Sequential(
        nn.ConvTranspose2d(in_channel, out_channel, kernel, stride),
        ConvTranspose2dSamePad(kernel, stride),
        nn.ReLU(True)
    )
    return layer


class InceptionBlock(nn.Module):
    def __init__(self, in_channel, out1, out2, out3):
        super(InceptionBlock, self).__init__()
        # 定义inception模块第一条线路，2*2卷积，步长为2
        self.branch2x2 = conv_relu(in_channel, out1, 2)

        # 定义inception模块第二条线路 3*3卷积，步长为2
        self.branch3x3 = conv_relu(in_channel, out2, 3)

        # 定义inception模块的第三条线路，5*5卷积，步长为2
        self.branch5x5 = conv_relu(in_channel, out3, 5)

    def forward(self, x):
        f1 = self.branch2x2(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        output = torch.cat((f1, f2, f3), dim=1)
        return output

class TransposeInceptionBlock(nn.Module):
    def __init__(self, in_channel, out1, out2, out3):
        super(TransposeInceptionBlock, self).__init__()
        # 定义inception模块第一条线路，2*2卷积，步长为2
        self.branch2x2 = transpose_conv_relu(in_channel, out1, 2)

        # 定义inception模块第二条线路 3*3卷积，步长为2
        self.branch3x3 = transpose_conv_relu(in_channel, out2, 3)

        # 定义inception模块的第三条线路，5*5卷积，步长为2
        self.branch5x5 = transpose_conv_relu(in_channel, out3, 5)

    def forward(self, x):
        f1 = self.branch2x2(x)
        f2 = self.branch3x3(x)
        f3 = self.branch5x5(x)
        output = torch.cat((f1, f2, f3), dim=1)
        return output
    
