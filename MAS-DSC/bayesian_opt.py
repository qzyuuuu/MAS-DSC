import torch
import scipy.io as sio
import numpy as np
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from main import RIABNet_Brain, RIABNet_Coil20, RIABNet_ORL, train
import os
import argparse
import warnings

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='DSCNet')
    parser.add_argument('--db', default='orl',
                        choices=['coil20', 'coil100', 'orl', 'reuters10k', 'stl', 'brain', 'detection'])
    parser.add_argument('--show-freq', default=10, type=int)
    parser.add_argument('--normalized', default=True)
    parser.add_argument('--use-prior-knowledge', default=False)
    parser.add_argument('--search-epochs', default=40)

    args = parser.parse_args()
    print(args)

    # 运行贝叶斯超参数优化的search的epoch，不要太大否则会占用太多的运行时间
    search_epochs = int(args.search_epochs)
    use_prior_knowledge = args.use_prior_knowledge.lower() == 'true'
    db = args.db
    if db == 'coil20':
        data = sio.loadmat('datasets/COIL20.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        num_sample = x.shape[0]
    elif db == 'coil100':
        data = sio.loadmat('datasets/COIL100.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        num_sample = x.shape[0]
    elif db == 'orl':
        data = sio.loadmat('datasets/ORL_32x32.mat')
        x, y = data['fea'].reshape((-1, 1, 32, 32)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        num_sample = x.shape[0]
    elif db == 'detection':
        data = sio.loadmat('datasets/detection.mat')
        x, y = data['fea'].reshape((-1, 3, 64, 64)), data['gnd']
        y = np.squeeze(y - 1)  # y in [0, 1, ..., K-1]
        # network and optimization parameters
        mean = np.mean(x, axis=(0, 1, 2), keepdims=True)
        std = np.std(x, axis=(0, 1, 2), keepdims=True)
        x = (x - mean) / std
        num_sample = x.shape[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if db == 'coil20':
        dscnet = RIABNet_Coil20(num_sample=num_sample)
    elif db == 'coil100':
        dscnet = RIABNet_ORL(num_sample=num_sample)
    elif db == 'orl':
        dscnet = RIABNet_ORL(num_sample=num_sample)
    elif db == 'brain':
        dscnet = RIABNet_Brain(num_sample=num_sample)
    elif db == 'detection':
        dscnet = RIABNet_Brain(num_sample=num_sample)
    dscnet.to(device)
    ae_state_dict = torch.load('pretrained_weights_new/%s.pkl' % db)
    dscnet.ae.load_state_dict(ae_state_dict)
    print("Pretrained ae weights are loaded successfully.")


    def cal_acc(weight_coef, weight_selfExp, weight_acc, alpha, dim_subspace, ro):
        cur_best_acc = train(dscnet, x, y, search_epochs, use_prior_knowledge=use_prior_knowledge,
                             is_print=False, weight_coef=weight_coef, weight_selfExp=weight_selfExp,
                             weight_acc=weight_acc,
                             alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=1000, device=device)
        return cur_best_acc


    def cal_acc_without_preknowledge(weight_coef, weight_selfExp, alpha, dim_subspace, ro):
        cur_best_acc = train(dscnet, x, y, search_epochs, use_prior_knowledge=use_prior_knowledge,
                             is_print=False, weight_coef=weight_coef, weight_selfExp=weight_selfExp, weight_acc=0,
                             alpha=alpha, dim_subspace=dim_subspace, ro=ro, show=1000, device=device)
        return cur_best_acc


    # 假设的损失函数
    def func(params):
        weight_coef = params['weight_coef']
        weight_selfExp = params['weight_selfExp']
        alpha = params['alpha']
        dim_subspace = int(params['dim_subspace'])
        ro = int(params['ro'])

        if use_prior_knowledge:
            weight_acc = params['weight_acc']
            # 这里应该是根据a和b计算得到的损失值，现在我们简单地返回a和b的和
            loss = cal_acc(weight_coef=weight_coef, weight_selfExp=weight_selfExp,
                           weight_acc=weight_acc, alpha=alpha, dim_subspace=dim_subspace, ro=ro)
        else:
            loss = cal_acc_without_preknowledge(weight_coef=weight_coef, weight_selfExp=weight_selfExp,
                                                alpha=alpha, dim_subspace=dim_subspace, ro=ro)
        return {'loss': -1 * loss, 'status': STATUS_OK}


    # 定义超参数空间
    # 可以指定是整数，我们使用hp.quniform并指定整数步长，q=1表示步长为1
    # 可以指定浮点数，我们使用hp.uniform定义其范围
    space = {
        'ro': hp.quniform('ro', 3, 12, 1),
        'dim_subspace': hp.quniform('dim_subspace', 3, 12, 1),
        'weight_coef': hp.uniform('weight_coef', 0.1, 4.0),
        'weight_selfExp': hp.uniform('weight_selfExp', 0.1, 2.0),
        'alpha': hp.uniform('alpha', 0.1, 0.5),
    }

    if use_prior_knowledge:
        space['weight_acc'] = hp.uniform('weight_acc', 0.1, 2.0)

    # 运行贝叶斯优化
    best = fmin(
        func,
        space=space,
        algo=tpe.suggest,
        max_evals=10
    )

    print(best)
