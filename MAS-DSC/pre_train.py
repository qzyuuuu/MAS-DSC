import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from conv import Conv2dSamePad, ConvTranspose2dSamePad
from inception import InceptionBlock, TransposeInceptionBlock
from main import RIABConvAE_Coil20, RIABConvAE_ORL, RIABConvAE_Brain
from post_clustering import spectral_clustering, acc, nmi
import scipy.io as sio
import math
from resnet import ResNetBlock, TransposeResNetBlock


def train(model, x, y, epochs, lr=1e-3,
          device='cuda', alpha=0.04, dim_subspace=12, ro=8, show=10):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.float32, device=device)
    x = x.to(device)
    if isinstance(y, torch.Tensor):
        y = y.to('cpu').numpy()
    K = len(np.unique(y))
    for epoch in range(epochs):
        x_recon = model(x)
        loss = model.loss_fn(x, x_recon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if epoch % show == 0 or epoch == epochs - 1:
            print('Epoch %02d: loss=%.4f' % (epoch, loss.item()))
    torch.save(model.state_dict(), args.save_dir + '/%s.pkl' % args.db)


if __name__ == "__main__":
    import argparse
    import warnings

    parser = argparse.ArgumentParser(description='RIAB-AE')
    parser.add_argument('--db', default='orl',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl', 'brain', 'detection'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--ae-weights', default=None)
    parser.add_argument('--save-dir', default='pretrained_weights_new')
    parser.add_argument('--normalized', default=True)

    args = parser.parse_args()
    print(args)
    import os

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        epochs = 1000
        weight_coef = 1.0
        weight_selfExp = 75

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
        epochs = 10000

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1 #
    elif db == 'brain':
        # load data
        data = sio.loadmat('datasets/brain.mat')
        x, y = data['fea'].reshape((-1, 3, 256, 256)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        if args.normalized == True:
            mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
            std = np.std(x, axis=(0, 1, 2), keepdims=True)
            x = (x - mean) / std
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 500

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 1  #
    elif db == 'detection':
        # load data
        data = sio.loadmat('datasets/detection.mat')
        x, y = data['fea'].reshape((-1, 3, 64, 64)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        if args.normalized == True:
            mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
            std = np.std(x, axis=(0, 1, 2), keepdims=True)
            x = (x - mean) / std
        # network and optimization parameters
        num_sample = x.shape[0]
        channels = [1, 3, 3, 5]
        kernels = [3, 3, 3]
        epochs = 1000

        # post clustering parameters
        alpha = 0.2  # threshold of C
        dim_subspace = 3  # dimension of each subspace
        ro = 2 #

    if db == 'coil20':
        riabConvAE = RIABConvAE_Coil20()
    elif db == 'coil100':
        riabConvAE = RIABConvAE_ORL()
    elif db == 'orl':
        riabConvAE = RIABConvAE_ORL()
    elif db == 'brain':
        riabConvAE = RIABConvAE_Brain()
    elif db == 'detection':
        riabConvAE = RIABConvAE_Brain()
    riabConvAE.to(device)

    # load the pretrained weights which are provided by the original author in
    # https://github.com/panji1990/Deep-subspace-clustering-networks
    # ae_state_dict = torch.load('pretrained_weights_original/%s.pkl' % db)
    # dscnet.ae.load_state_dict(ae_state_dict)
    # print("Pretrained ae weights are loaded successfully.")

    train(riabConvAE, x, y, epochs, alpha=alpha,
          dim_subspace=dim_subspace, ro=ro, show=args.show_freq, device=device)
    torch.save(riabConvAE.state_dict(), args.save_dir + '/%s-model.ckp' % args.db)


