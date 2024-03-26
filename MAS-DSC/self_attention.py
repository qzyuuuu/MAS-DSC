import torch
import torch.nn as nn
import math
import sys
sys.path.append('.')

# 定义类，继承Module
class ECA_Block(nn.Module):
# 初始化，根据通道数计算卷积核大小，gamma默认是2
	def __init__(self, channel, b=1, gamma=2):
		super(ECA_Block, self).__init__()
# 使用Log函数按照输入通道数自适应的计算卷积核大小
		kernel_size = int(abs((math.log(channel, 2) + b) / gamma))
		kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1


# 全局平均池化
		self.avg_pool = nn.AdaptiveAvgPool2d(1)
# 1d卷积通常是在序列模型中使用，1, 1 表示每一个step上的值长度都只有1
		self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
		self.sigmoid = nn.Sigmoid()


# 前传，一般来讲要计算x的size，b, c, h, w = x.size
	def forward(self, x):
# 或者用view([b, 1, c])，调整y到序列形式，1表示每一个step上的特征长度
		y = self.avg_pool(x)
		y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
		y = self.sigmoid(y)
		return x * y.expand_as(x)
