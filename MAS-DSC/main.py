"""
By Xifeng Guo (guoxifeng1990@163.com), May 13, 2020.
All rights reserved.
"""
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from conv import Conv2dSamePad, ConvTranspose2dSamePad
from inception import InceptionBlock, TransposeInceptionBlock
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math
from predict_by_svm import pre_pred_by_svm
from resnet import ResNetBlock, TransposeResNetBlock
from self_attention import ECA_Block
import json
import shutil
import joblib

class ConvAE(nn.Module):
    def __init__(self, channels, kernels):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(ConvAE, self).__init__()
        assert isinstance(channels, list) and isinstance(kernels, list)
        self.encoder = nn.Sequential()
        for i in range(1, len(channels)):
            #  Each layer will divide the size of feature map by 2
            self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernels[i - 1], 2))
            self.encoder.add_module('conv%d' % i,
                                    nn.Conv2d(channels[i - 1], channels[i], kernel_size=kernels[i - 1], stride=2))
            self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        self.decoder = nn.Sequential()
        channels = list(reversed(channels))
        kernels = list(reversed(kernels))
        for i in range(len(channels) - 1):
            # Each layer will double the size of feature map
            self.decoder.add_module('deconv%d' % (i + 1),
                                    nn.ConvTranspose2d(channels[i], channels[i + 1], kernel_size=kernels[i], stride=2))
            self.decoder.add_module('padd%d' % i, ConvTranspose2dSamePad(kernels[i], 2))
            self.decoder.add_module('relud%d' % i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y

class SelfExpression(nn.Module):
    def __init__(self, n):
        super(SelfExpression, self).__init__()
        self.Coefficient = nn.Parameter(1.0e-8 * torch.ones(n, n, dtype=torch.float32), requires_grad=True)

    def forward(self, x):  # shape=[n, d]
        y = torch.matmul(self.Coefficient, x)
        # 根据半监督的信息
        return y

class DSCNet(nn.Module):
    def __init__(self, channels, kernels, num_sample):
        super(DSCNet, self).__init__()
        self.n = num_sample
        self.ae = ConvAE(channels, kernels)
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, y, y_pred, weight_coef, weight_selfExp, weight_acc, use_prior_knowledge):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        if use_prior_knowledge:
            loss_acc = np.sum((y - y_pred)**2) 
            loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp + weight_acc * loss_acc
        else:
            loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        return loss

# Resnet and Inception Architecuture Based Conv AutoEnoder(RIABConvAE)
class RIABConvAE_ORL(nn.Module):
    def __init__(self):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(RIABConvAE_ORL, self).__init__()
        # 构造encoder结构
        self.encoder = nn.Sequential()    

        channels = [1, 3, 3, 3, 5]
        i = 1
        #layer 1 普通卷积层，输入32*32*1, 输出 16*16*3，卷积核大小3*3，步长为2
        in_channel, out_channel = 1, 3
        kernel, stride = 3, 2
        self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernel, 2))
        self.encoder.add_module('conv%d' % i,
                                nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=2))
        self.encoder.add_module('relu%d' % i, nn.ReLU(True))
        
        i = 2
        #layer2 inception层，输入16*16*3，输出8*8*3
        in_channel = 3
        out_channel1, out_channel2, out_channel3 = 1, 1, 1
        inception_layer = InceptionBlock(in_channel, out_channel1, out_channel2, out_channel3)
        self.encoder.add_module('inception%d' %i, inception_layer)

        #encoder 增加自注意力机制
        input_channel = 3
        eca_layer = ECA_Block(input_channel)
        self.encoder.add_module('eca_block%', eca_layer)

        i = 3
        #layer3 resnet层，输入8*8*3，输出8*8*3
        in_channel, out_channel = 3, 3
        kernel1, kernel2 = 2, 3
        resnet_layer = ResNetBlock(in_channel, out_channel, kernel1, kernel2)
        self.encoder.add_module('resnet%d' %i, resnet_layer)

        i = 4
        #layer4 普通层，输入8*8*3, 输出4*4*5，卷积核大小3*3，步长为2
        in_channel, out_channel = 3, 5
        kernel, stride = 3, 2
        self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernel, stride))
        self.encoder.add_module('conv%d' % i,
                                nn.Conv2d(in_channel, out_channel, kernel, stride))
        self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        # 构造dencoder结构
        self.decoder = nn.Sequential()    

        de_i = 1
        #反layer1 普通层，输入4*4*5, 输出8*8*3，卷积核大小3*3，步长为2
        in_channel, out_channel = 5, 3
        kernel, stride = 3, 2
        self.decoder.add_module('deconv%d' % de_i,
                                    nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.decoder.add_module('padd%d' % de_i, ConvTranspose2dSamePad(kernel, 2))
        self.decoder.add_module('relud%d' % de_i, nn.ReLU(True))

        de_i = 2 
        #反layer2 resnet，输入8*8*3, 输出8*8*3
        in_channel, out_channel = 3, 3
        kernel1, kernel2 = 3, 2
        resnet_layer = TransposeResNetBlock(in_channel, out_channel, kernel1, kernel2)
        self.decoder.add_module('de_resnet%d' %de_i, resnet_layer)

        de_i = 3
        #反layer3 inception，输入8*8*3，输出16*16*3
        in_channel = 3
        out_channel1, out_channel2, out_channel3 = 1, 1, 1
        inception_layer = TransposeInceptionBlock(in_channel, out_channel1, out_channel2, out_channel3)
        self.decoder.add_module('de_inception%d' %de_i, inception_layer)

        #decoder 增加自注意力机制
        input_channel = 3
        eca_layer = ECA_Block(input_channel)
        self.decoder.add_module('eca_block%', eca_layer)

        de_i = 4
        #反layer 4 普通卷积层，输入16*16*3, 输出 32*32*1，卷积核大小3*3，步长为2
        in_channel, out_channel = 3, 1
        kernel, stride = 3, 2
        self.decoder.add_module('deconv%d' % de_i,
                                    nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.decoder.add_module('padd%d' % de_i, ConvTranspose2dSamePad(kernel, 2))
        self.decoder.add_module('relud%d' % de_i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y
    
    def loss_fn(self, x, x_recon):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        return loss_ae

class RIABNet_ORL(nn.Module):
    def __init__(self, num_sample):
        super(RIABNet_ORL, self).__init__()
        self.n = num_sample
        self.ae = RIABConvAE_ORL()
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, y, y_pred, weight_coef, weight_selfExp, weight_acc, use_prior_knowledge):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        if use_prior_knowledge:
            loss_acc = np.sum((y - y_pred)**2) 
            loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp + weight_acc * loss_acc
        else:
            loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        return loss

class RIABConvAE_Coil20(nn.Module):
    def __init__(self):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(RIABConvAE_Coil20, self).__init__()
        # 构造encoder结构
        self.encoder = nn.Sequential()    

        channels = [1, 6, 6, 6, 10]
        i = 1
        #layer 1 普通卷积层，输入32*32*1, 输出 16*16*6，卷积核大小3*3，步长为2
        in_channel, out_channel = 1, 6
        kernel, stride = 3, 2
        self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernel, 2))
        self.encoder.add_module('conv%d' % i,
                                nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=2))
        self.encoder.add_module('relu%d' % i, nn.ReLU(True))
        
        i = 2
        #layer2 inception层，输入16*16*6，输出8*8*6
        in_channel = 6
        out_channel1, out_channel2, out_channel3 = 2, 2, 2
        inception_layer = InceptionBlock(in_channel, out_channel1, out_channel2, out_channel3)
        self.encoder.add_module('inception%d' %i, inception_layer)

        #encoder 增加自注意力机制
        input_channel = 6
        eca_layer = ECA_Block(input_channel)
        self.encoder.add_module('eca_block%', eca_layer)

        i = 3
        #layer3 resnet层，输入8*8*6，输出8*8*6
        in_channel, out_channel = 6, 6
        kernel1, kernel2 = 2, 3
        resnet_layer = ResNetBlock(in_channel, out_channel, kernel1, kernel2)
        self.encoder.add_module('resnet%d' %i, resnet_layer)

        i = 4
        #layer4 普通层，输入8*8*6, 输出4*4*15，卷积核大小3*3，步长为2
        in_channel, out_channel = 6, 15
        kernel, stride = 3, 2
        self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernel, stride))
        self.encoder.add_module('conv%d' % i,
                                nn.Conv2d(in_channel, out_channel, kernel, stride))
        self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        # 构造dencoder结构
        self.decoder = nn.Sequential()    

        de_i = 1
        #反layer1 普通层，输入4*4*15, 输出8*8*6，卷积核大小3*3，步长为2
        in_channel, out_channel = 15, 6
        kernel, stride = 3, 2
        self.decoder.add_module('deconv%d' % de_i,
                                    nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.decoder.add_module('padd%d' % de_i, ConvTranspose2dSamePad(kernel, 2))
        self.decoder.add_module('relud%d' % de_i, nn.ReLU(True))

        de_i = 2 
        #反layer2 resnet，输入8*8*6, 输出8*8*6
        in_channel, out_channel = 6, 6
        kernel1, kernel2 = 3, 2
        resnet_layer = TransposeResNetBlock(in_channel, out_channel, kernel1, kernel2)
        self.decoder.add_module('de_resnet%d' %de_i, resnet_layer)

        de_i = 3
        #反layer3 inception，输入8*8*6，输出16*16*6
        in_channel = 6
        out_channel1, out_channel2, out_channel3 = 2, 2, 2
        inception_layer = TransposeInceptionBlock(in_channel, out_channel1, out_channel2, out_channel3)
        self.decoder.add_module('de_inception%d' %de_i, inception_layer)

        #decoder 增加自注意力机制
        input_channel = 3
        eca_layer = ECA_Block(input_channel)
        self.decoder.add_module('eca_block%', eca_layer)

        de_i = 4
        #反layer 4 普通卷积层，输入16*16*6, 输出 32*32*1，卷积核大小3*3，步长为2
        in_channel, out_channel = 6, 1
        kernel, stride = 3, 2
        self.decoder.add_module('deconv%d' % de_i,
                                    nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.decoder.add_module('padd%d' % de_i, ConvTranspose2dSamePad(kernel, 2))
        self.decoder.add_module('relud%d' % de_i, nn.ReLU(True))

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y
    
    def loss_fn(self, x, x_recon):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        return loss_ae

class RIABNet_Coil20(nn.Module):
    def __init__(self, num_sample):
        super(RIABNet_Coil20, self).__init__()
        self.n = num_sample
        self.ae = RIABConvAE_Coil20()
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, y, y_pred, weight_coef, weight_selfExp, weight_acc, use_prior_knowledge):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        if use_prior_knowledge:
            loss_acc = np.sum((y - y_pred)**2) 
            loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp + weight_acc * loss_acc
        else:
            loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        return loss

class RIABConvAE_Brain(nn.Module):
    def __init__(self):
        """
        :param channels: a list containing all channels including the input image channel (1 for gray, 3 for RGB)
        :param kernels:  a list containing all kernel sizes, it should satisfy: len(kernels) = len(channels) - 1.
        """
        super(RIABConvAE_Brain, self).__init__()
        # 定义标准化时使用的均值和标准差

        # 构造encoder结构
        self.encoder = nn.Sequential()  
      
        i = 1
        #layer 1 普通卷积层，输入32*32*1, 输出 16*16*6，卷积核大小3*3，步长为2
        in_channel, out_channel = 3, 6
        kernel, stride = 3, 2
        self.encoder.add_module('1pad%d' % i, Conv2dSamePad(kernel, 2))
        self.encoder.add_module('1conv%d' % i,
                                nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=2))
        self.encoder.add_module('1relu%d' % i, nn.ReLU(True))
        
        in_channel, out_channel = 6, 12
        kernel, stride = 3, 2
        self.encoder.add_module('2pad%d' % i, Conv2dSamePad(kernel, 2))
        self.encoder.add_module('2conv%d' % i,
                                nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=2))
        self.encoder.add_module('2relu%d' % i, nn.ReLU(True))

        in_channel, out_channel = 12, 24
        kernel, stride = 3, 2
        self.encoder.add_module('3pad%d' % i, Conv2dSamePad(kernel, 2))
        self.encoder.add_module('3conv%d' % i,
                                nn.Conv2d(in_channel, out_channel, kernel_size=kernel, stride=2))
        self.encoder.add_module('3relu%d' % i, nn.ReLU(True))

        i = 2
        #layer2 inception层，输入16*16*6，输出8*8*6
        in_channel = 24
        out_channel1, out_channel2, out_channel3 = 8, 8, 8
        inception_layer = InceptionBlock(in_channel, out_channel1, out_channel2, out_channel3)
        self.encoder.add_module('inception%d' %i, inception_layer)

        #encoder 增加自注意力机制
        input_channel = 24
        eca_layer = ECA_Block(input_channel)
        self.encoder.add_module('eca_block%', eca_layer)

        i = 3
        #layer3 resnet层，输入8*8*6，输出8*8*6
        in_channel, out_channel = 24, 24
        kernel1, kernel2 = 2, 3
        resnet_layer = ResNetBlock(in_channel, out_channel, kernel1, kernel2)
        self.encoder.add_module('resnet%d' %i, resnet_layer)

        i = 4
        #layer4 普通层，输入8*8*6, 输出4*4*15，卷积核大小3*3，步长为2
        in_channel, out_channel = 24, 16
        kernel, stride = 3, 2
        self.encoder.add_module('pad%d' % i, Conv2dSamePad(kernel, stride))
        self.encoder.add_module('conv%d' % i,
                                nn.Conv2d(in_channel, out_channel, kernel, stride))
        self.encoder.add_module('relu%d' % i, nn.ReLU(True))

        # 构造dencoder结构
        self.decoder = nn.Sequential()    

        de_i = 1
        #反layer1 普通层，输入4*4*15, 输出8*8*6，卷积核大小3*3，步长为2
        in_channel, out_channel = 16, 24
        kernel, stride = 3, 2
        self.decoder.add_module('deconv%d' % de_i,
                                    nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.decoder.add_module('padd%d' % de_i, ConvTranspose2dSamePad(kernel, 2))
        self.decoder.add_module('relud%d' % de_i, nn.ReLU(True))

        de_i = 2 
        #反layer2 resnet，输入8*8*6, 输出8*8*6
        in_channel, out_channel = 24, 24
        kernel1, kernel2 = 3, 2
        resnet_layer = TransposeResNetBlock(in_channel, out_channel, kernel1, kernel2)
        self.decoder.add_module('de_resnet%d' %de_i, resnet_layer)

        de_i = 3
        #反layer3 inception，输入8*8*6，输出16*16*6
        in_channel = 24
        out_channel1, out_channel2, out_channel3 = 8, 8, 8
        inception_layer = TransposeInceptionBlock(in_channel, out_channel1, out_channel2, out_channel3)
        self.decoder.add_module('de_inception%d' %de_i, inception_layer)

        #decoder 增加自注意力机制
        input_channel = 24
        eca_layer = ECA_Block(input_channel)
        self.decoder.add_module('eca_block%', eca_layer)

        de_i = 4
        #反layer 4 普通卷积层，输入16*16*6, 输出 32*32*3，卷积核大小3*3，步长为2
        in_channel, out_channel = 24, 12
        kernel, stride = 3, 2
        self.decoder.add_module('1deconv%d' % de_i,
                                    nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.decoder.add_module('1padd%d' % de_i, ConvTranspose2dSamePad(kernel, 2))
        self.decoder.add_module('1relud%d' % de_i, nn.ReLU(True))

        in_channel, out_channel = 12, 6
        kernel, stride = 3, 2
        self.decoder.add_module('2deconv%d' % de_i,
                                    nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.decoder.add_module('2padd%d' % de_i, ConvTranspose2dSamePad(kernel, 2))
        self.decoder.add_module('2relud%d' % de_i, nn.ReLU(True))

        in_channel, out_channel = 6, 3
        kernel, stride = 3, 2
        self.decoder.add_module('3deconv%d' % de_i,
                                    nn.ConvTranspose2d(in_channel, out_channel, kernel, stride))
        self.decoder.add_module('3padd%d' % de_i, ConvTranspose2dSamePad(kernel, 2))
        self.decoder.add_module('3relud%d' % de_i, nn.ReLU(True))

        #self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                init.constant_(m.bias, 0)

    def forward(self, x):
        h = self.encoder(x)
        y = self.decoder(h)
        return y
    
    def loss_fn(self, x, x_recon):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        return loss_ae

class RIABNet_Brain(nn.Module):
    def __init__(self, num_sample):
        super(RIABNet_Brain, self).__init__()
        self.n = num_sample
        self.ae = RIABConvAE_Brain()
        self.self_expression = SelfExpression(self.n)

    def forward(self, x):  # shape=[n, c, w, h]
        z = self.ae.encoder(x)

        # self expression layer, reshape to vectors, multiply Coefficient, then reshape back
        shape = z.shape
        z = z.view(self.n, -1)  # shape=[n, d]
        z_recon = self.self_expression(z)  # shape=[n, d]
        z_recon_reshape = z_recon.view(shape)

        x_recon = self.ae.decoder(z_recon_reshape)  # shape=[n, c, w, h]
        return x_recon, z, z_recon

    def loss_fn(self, x, x_recon, z, z_recon, y, y_pred, weight_coef, weight_selfExp, weight_acc, use_prior_knowledge):
        loss_ae = F.mse_loss(x_recon, x, reduction='sum')
        loss_coef = torch.sum(torch.pow(self.self_expression.Coefficient, 2))
        loss_selfExp = F.mse_loss(z_recon, z, reduction='sum')
        if use_prior_knowledge:
            loss_acc = acc(y, y_pred)
            #loss_acc = np.sum((y - y_pred)**2) 
            loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp + weight_acc * loss_acc
        else:
            loss = loss_ae + weight_coef * loss_coef + weight_selfExp * loss_selfExp
        return loss
      
# 调整学习率
def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def train(model,  # type: DSCNet
          x, y, epochs, use_prior_knowledge, is_print, lr=1e-3, weight_coef=1.0, weight_selfExp=150, weight_acc = 10, device='cuda',
          alpha=0.04, dim_subspace=12, ro=8, show=10, threshold = 0.7):
    best_acc = 0.0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))

    # 这里增加svm的相关的先验知识
    if use_prior_knowledge:
        clf = joblib.load('svm_model.joblib')
        y_pred = pre_pred_by_svm('svm_model.joblib', 'datasets/detection.mat', threshold)
    for epoch in range(epochs):
        if epoch > 100:
            set_learning_rate(optimizer=optimizer, lr=1e-4)
        # if use_prior_knowledge:
        #     C = model.self_expression.Coefficient.detach().to('cpu').numpy()
        #     r = dim_subspace * K + 1
        #     if r > min(C.shape):
        #         return 0
        #     y_pred  = spectral_clustering(C, K, dim_subspace, alpha, ro)
        x_recon, z, z_recon = model(x)
        loss = model.loss_fn(x, x_recon, z, z_recon, y, y_pred, weight_coef, weight_selfExp, weight_acc, use_prior_knowledge)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            C = model.self_expression.Coefficient.detach().to('cpu').numpy()
            r = dim_subspace * K + 1
            if r > min(C.shape):
                return 0
            y_pred = spectral_clustering(C, K, dim_subspace, alpha, ro)
            epoch_acc = acc(y, y_pred)
            if epoch_acc > best_acc:
                best_acc = epoch_acc
            if is_print:
                print('Epoch %02d: loss=%.4f, acc=%.4f, nmi=%.4f' %
                  (epoch, loss.item() / y_pred.shape[0], epoch_acc, nmi(y, y_pred)))
    return best_acc
        



if __name__ == "__main__":
    # for test
    # input = torch.randn(1, 1, 32, 32)
    # dscnet = RIABNet(num_sample=1)
    # print(dscnet)


    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='orl',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl', 'brain', 'detection'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='results')
    parser.add_argument('--normalized', default=True)
    parser.add_argument('--use-prior-knowledge', default=True)
    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    threshold = 0.7
    idx = []
    db = args.db
    if db == 'coil20':
        # load data
        data = sio.loadmat('datasets/COIL20.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 15]
        kernels = [3]
        epochs = 500
        weight_coef = 1.0
        weight_selfExp = 75
        weight_acc = 0.1

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
        warnings.warn("You can uncomment line#64 in post_clustering.py to get better result for this dataset!")
    elif db == 'coil100':
        # load data
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]

        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 50]
        kernels = [5]
        epochs = 120
        weight_coef = 1.0
        weight_selfExp = 15
        weight_acc = 0.1

        # post clustering parameters
        alpha = 0.04  # threshold of C
        dim_subspace = 12  # dimension of each subspace
        ro = 8  #
    elif db == 'orl':
        # load data
        data = sio.loadmat('datasets/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 500
        weight_coef = 2.0
        weight_selfExp = 1.3876
        weight_acc = 0.658

        # post clustering parameters
        alpha = 0.1115  # threshold of C
        dim_subspace = 6  # dimension of each subspace
        ro = 6  #
    elif db == "brain":
        # load data
        data = sio.loadmat('datasets/brain.mat')
        x, y = data['fea'].reshape((-1, 3, 256, 256)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # network and optimization parameters
        if args.normalized == True:
            mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
            std = np.std(x, axis=(0, 1, 2), keepdims=True)
            x = (x - mean) / std
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 300
        weight_coef = 2.0
        weight_selfExp = 0.2
        weight_acc = 0.2

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 8  #
    elif db == "detection":
        # load data
        data = sio.loadmat('datasets/detection.mat')
        x, y = data['fea'].reshape((-1, 3, 64, 64)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        idx = data['idx']
        idx = np.squeeze(idx)
        # network and optimization parameters
        if args.normalized == True:
            mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
            std = np.std(x, axis=(0, 1, 2), keepdims=True)
            x = (x - mean) / std
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 100
        weight_coef = 3.9202926106100815 #2.0
        weight_selfExp = 0.24661993615951372 #0.2
        weight_acc = 1.7017943084025926 #0.2

        # post clustering parameters
        alpha =  0.17311665754681443 #0.2  # threshold of C
        dim_subspace = 8 #12  # dimension of each subspace
        ro = 8 #4  
        # 贝叶斯优化搜索最优解
        threshold = 0.63

    dscnet = DSCNet(num_sample=num_sample, channels=channels, kernels=kernels)
    if db == 'coil20':
        dscnet = RIABNet_Coil20(num_sample=num_sample)
    elif db == 'coil100':
        dscnet = RIABNet_ORL(num_sample=num_sample)
    elif db == 'orl':
        dscnet = RIABNet_ORL(num_sample=num_sample)
    elif db =='brain':
         dscnet = RIABNet_Brain(num_sample=num_sample)
    elif db =='detection':
        dscnet = RIABNet_Brain(num_sample=num_sample)
        
    dscnet.to(device)

    # load the pretrained weights which are provided by the original author in
    # https://github.com/panji1990/Deep-subspace-clustering-networks
    ae_state_dict = torch.load('pretrained_weights_new/%s.pkl' % db)
    dscnet.ae.load_state_dict(ae_state_dict)
    print("Pretrained ae weights are loaded successfully.")
    cur_best_acc = train(dscnet, x, y, epochs, use_prior_knowledge=args.use_prior_knowledge.lower()== 'true', is_print = True, weight_coef=weight_coef, weight_selfExp=weight_selfExp, weight_acc=weight_acc,
          alpha=alpha, dim_subspace=dim_subspace, ro=ro, threshold = threshold, show=args.show_freq, device=device)
    torch.save(dscnet.state_dict(), args.save_dir + '/%s-model.ckp' % args.db)

    # 和最优的acc对比，保存最好的结果
    try:
        filePath = os.path.join('pretrained_weights_best', args.db, args.db + '.json')
        with open(filePath, 'r') as file:
            data = json.load(file)
            old_best_acc = data.get('best_acc', 0.0)
        if cur_best_acc > old_best_acc: 
            with open(filePath, 'w') as file:
                data['best_acc'] = cur_best_acc
                data['weight_coef'] = weight_coef
                data['weight_selfExp'] = weight_selfExp
                data['weight_acc'] = weight_acc
                data['alpha'] = alpha
                data['dim_subspace'] = dim_subspace
                data['ro'] = ro
                json.dump(data, file, indent=4)
                print('更新了best_acc的值为', cur_best_acc)
                source_path = os.path.join('pretrained_weights_new', args.db + '.pkl')
                dest_path = os.path.join('pretrained_weights_best', args.db, args.db + '.pkl')
                shutil.copyfile(source_path, dest_path)
    except IOError as e:
        print('无法打开或写入文件', e)
    except json.JSONDecodeError as e:
        print('文件不是有效的JSON格式：', e)
    except Exception as e:
        print('发生了意外的错误：', e)

        

