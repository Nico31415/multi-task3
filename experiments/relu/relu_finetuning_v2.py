from copy import deepcopy
import argparse
import sys
import os
from pathlib import Path
sys.path.append('')
import math

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from sklearn.linear_model import Ridge

import functions.networks as nt

def f1_norm(model):
    return (model.scaling*torch.sum(
        torch.sqrt(torch.sum(model.module._modules['features'][0].weight**2, dim=1))*torch.abs(model.module._modules['readout'].weight[-1])
    )).item()

def gs_norm(model):
    return (model.scaling*torch.sum(
        torch.sqrt(torch.sum(model.module._modules['features'][0].weight**2, dim=1))*\
        torch.sqrt(torch.sum(model.module._modules['readout'].weight**2, dim=0))
    )).item()

def train_two_tasks(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    x, y, task = train_data
    val_x, val_y1, val_y2 = val_data
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(x)
        pred = select_output(pred, task)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                new_df = pd.DataFrame({
                    'loss': [
                        F.mse_loss(model(val_x)[:,0], val_y1).item(),
                        F.mse_loss(model(val_x)[:,1], val_y2).item()
                    ],
                    'split': ['val_1', 'val_2']
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train_two_tasks(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold)
    with torch.no_grad():
        new_df = pd.DataFrame({
            'loss': [
                F.mse_loss(model(val_x)[:,0], val_y1).item(),
                F.mse_loss(model(val_x)[:,1], val_y2).item()
            ],
            'split': ['val_1', 'val_2']
        })
        new_df['epoch'] = i
        test_preds.append(new_df)
    losses = pd.DataFrame({
        'epoch': np.arange(len(losses)),
        'loss': torch.stack(losses).numpy()
    })
    losses['split'] = 'train'
    test_preds = pd.concat(test_preds).reset_index(drop=True)
    df_norm = pd.DataFrame({
        'norm': ['F1', 'GS'],
        'value': [f1_norm(model), gs_norm(model)],
        'kind': 'student'
    })
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), df_norm

def train_one_task(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    x, y = train_data
    val_x, val_y = val_data
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(x)[:,0]
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                new_df = pd.DataFrame({
                    'loss': [
                        F.mse_loss(model(val_x)[:,0], val_y).item()
                    ],
                    'split': ['val']
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train_one_task(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold)
    with torch.no_grad():
        new_df = pd.DataFrame({
            'loss': [
                F.mse_loss(model(val_x)[:,0], val_y).item()
            ],
            'split': ['val']
        })
        new_df['epoch'] = i
        test_preds.append(new_df)
    losses = pd.DataFrame({
        'epoch': np.arange(len(losses)),
        'loss': torch.stack(losses).numpy()
    })
    losses['split'] = 'train'
    test_preds = pd.concat(test_preds).reset_index(drop=True)
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), f1_norm(model)

def relu(x, W, V):
    outp = x@W.T
    outp = torch.relu(outp)
    outp = V*outp
    return outp

def circular_sample(n_samples, inp_dim, variance=None):
    if variance is None:
        variance = torch.tensor([1.]*inp_dim)
    W = Normal(torch.tensor([0.]*inp_dim),torch.tensor(variance)).sample((n_samples,))
    return W/torch.sqrt(torch.mean(W**2, dim=-1, keepdims=True))

def sample_two_teachers(shape, seed, correlation=[1.]*6):
    W = ortho_group.rvs(shape[1], random_state=np.random.RandomState(seed=seed))
    W = torch.from_numpy(W).float()
    V = (torch.rand(shape[0])-0.5).sign()
    a = torch.tensor(correlation)
    b = torch.sqrt(1-a**2)
    W2 = a.unsqueeze(1)*W[:shape[0]]+b.unsqueeze(1)*W[shape[0]:(2*shape[0])]
    W = W[:shape[0]]
    V2 = (torch.rand(shape[0])-0.5).sign()
    return (W, V), (W2, V2)

def select_output(outp, task):
    task_oh = F.one_hot(task, outp.shape[1])
    return (outp*task_oh).sum(dim=-1)

def main(args):
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    param1, param2 = sample_two_teachers((args.n_units, args.inp_dim), args.seed, correlation=args.correlation)
    x1 = circular_sample(args.n_train1, args.inp_dim)
    x2 = circular_sample(args.n_train2, args.inp_dim, variance=args.train_var)
    val_x = circular_sample(10000, args.inp_dim)
    y1 = relu(x1, *param1).mean(dim=-1)
    y2 = relu(x2, *param2).mean(dim=-1)
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    task = torch.tensor([0]*args.n_train1+[1]*args.n_train2)
    val_y1 = relu(val_x, *param1).mean(dim=-1)
    val_y2 = relu(val_x, *param2).mean(dim=-1)
    if args.load_model:
        net = nt.ZeroOutput(nt.DenseNet(args.inp_dim, [1000], outp_dim=1), scaling=args.model_scaling, subtract=False)
        net.load_state_dict(torch.load(args.model_path))
        nn.init.kaiming_normal_(net.module.readout.weight)
        net.scaling = args.scaling
    else:
        outp_dim = 1 if args.one_task else 2
        net = nt.ZeroOutput(nt.DenseNet(args.inp_dim, [1000], outp_dim=outp_dim), scaling=args.scaling, subtract=False)
    if args.one_task:
        if args.setup in ['linear_readout', 'ntk']:
            if args.setup == 'ntk':
                h = net.module.ntk_features(x2)
                val_h = net.module.ntk_features(val_x)
            with torch.no_grad():
                if args.setup == 'linear_readout':
                    h = net.module.features(x2)
                    val_h = net.module.features(val_x)
            losses = []
            train_losses = []
            alphas = np.logspace(-5, 10, 100)
            for alpha in alphas:
                ridge_model = Ridge(alpha=alpha)
                ridge_model.fit(h, y2)
                losses.append(((ridge_model.predict(val_h)-val_y2.numpy())**2).mean())
                train_losses.append(((ridge_model.predict(h)-y2.numpy())**2).mean())
            train_df = pd.DataFrame({
                'alpha': alphas,
                'loss': train_losses
            })
            train_df['split'] = 'train'
            val_df = pd.DataFrame({
                'alpha': alphas,
                'loss': losses
            })
            val_df['split'] = 'val'
            df = pd.concat([train_df, val_df]).reset_index(drop=True)
        else:
            df, norm = train_one_task(net, (x2, y2), (val_x, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold)
            with torch.no_grad():
                net = nt.ZeroOutput(nt.DenseNet(args.inp_dim, [1000], outp_dim=1), scaling=args.model_scaling, subtract=False)
                net.load_state_dict(torch.load(args.model_path))
                gs_norm = norm + f1_norm(net)
            n_equal = torch.sum((torch.tensor(args.correlation)==1.).float())
            df_norm = pd.DataFrame({
                'norm': ['F1', 'GS', 'F1', 'GS'],
                'kind': ['student', 'student', 'teacher', 'teacher'],
                'value': [norm, gs_norm, args.n_units, (2*(args.n_units-n_equal)+torch.sqrt(torch.tensor(2.))*n_equal).item()]
            })
            df_norm.to_feather(os.path.join(args.save_path, 'df_norm.feather'))
    else:
        df, df_norm = train_two_tasks(net, (x, y, task), (val_x, val_y1, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold)
        with torch.no_grad():
            n_equal = torch.sum((torch.tensor(args.correlation)==1.).float())
            l1_norm = args.n_units
            gs_norm = (2*(args.n_units-n_equal)+torch.sqrt(torch.tensor(2.))*n_equal).item()
        df_norm = pd.concat([
            df_norm,
            pd.DataFrame({
                'norm': ['F1', 'GS'],
                'value': [l1_norm, gs_norm],
                'kind': 'teacher'
            })
        ]).reset_index(drop=True)
        df_norm.to_feather(os.path.join(args.save_path, 'df_norm.feather'))
    df.to_feather(os.path.join(args.save_path, 'df.feather'))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_scaling', type=float, default=1.)
    parser.add_argument('--n_train1', type=int, default=50)
    parser.add_argument('--n_train2', type=int, default=50)
    parser.add_argument('--n_units', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--no_tuning', action='store_true')
    parser.add_argument('--lr', type=float, default=1e20)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--scaling', type=float, default=1.)
    parser.add_argument('--correlation', type=float, default=[1.]*6, nargs='+')
    parser.add_argument('--train_var', type=float, default=[1.]*15, nargs='+')
    parser.add_argument('--one_task', action='store_true')
    parser.add_argument('--setup', choices=['backprop', 'ntk', 'linear_readout'], default='backprop')
    parser.add_argument('--inp_dim', default=15, type=int)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
