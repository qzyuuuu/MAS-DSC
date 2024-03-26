import torch
import torch.nn as nn
import torch.nn.functional as F

from conv import Conv2dSamePad, ConvTranspose2dSamePad
import sys
sys.path.append('.')#添加当前目录


class ResNetBlock(nn.Module):#定义残差块
    def __init__(self, in_channel, out_channel, kernel1, kernel2, stride=1):
        super(ResNetBlock, self).__init__()
        self.left = nn.Sequential(
            Conv2dSamePad(kernel1, stride),#卷积核大小为kernel1，步长为stride，padding为自动填充
            nn.Conv2d(in_channel, out_channel, kernel1, stride),
            nn.ReLU(True),

            Conv2dSamePad(kernel2, stride),
            nn.Conv2d(in_channel, out_channel, kernel2, stride),
            nn.ReLU(True)
        )

        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out

class TransposeResNetBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel1, kernel2, stride=1):
        super(TransposeResNetBlock, self).__init__()
        self.left = nn.Sequential(
            nn.ConvTranspose2d(in_channel, out_channel, kernel1, stride),
            ConvTranspose2dSamePad(kernel1, stride),
            nn.ReLU(True),

            nn.ConvTranspose2d(in_channel, out_channel, kernel2, stride),
            ConvTranspose2dSamePad(kernel2, stride),
            nn.ReLU(True)
        )

        self.shortcut = nn.Sequential()

    def forward(self, x):
        out = self.left(x)
        out = out + self.shortcut(x)
        out = F.relu(out)
        return out
