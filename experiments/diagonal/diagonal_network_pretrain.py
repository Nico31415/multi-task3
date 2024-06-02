from copy import deepcopy
import argparse
import math
import sys
import os
from pathlib import Path
sys.path.append('')

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
import pandas as pd

import functions.networks as nt

class DiagonalNet(nn.Module):
    def __init__(self, inp_dim, scaling=1.):
        super().__init__()
        self.w_pos = nn.Parameter(scaling*torch.ones(inp_dim))
        self.v_pos = nn.Parameter(scaling*torch.ones(inp_dim))
        self.v_neg = nn.Parameter(scaling*torch.ones(inp_dim))
        self.w_neg = nn.Parameter(scaling*torch.ones(inp_dim))
    
    def beta(self):
        return self.w_pos*self.v_pos-self.w_neg*self.v_neg

    def forward(self, x):
        return x@self.beta()

def l1_norm(x):
    return torch.sum(torch.abs(x)).item()

def l2_norm(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2)).item()

def train(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    norms = []
    x, y = train_data
    val_x, val_y = val_data
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                new_df = pd.DataFrame({
                    'loss': [F.mse_loss(model(val_x), val_y).item()]
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
                norms.append(
                    pd.DataFrame({
                        'norm': ['l1', 'l2'],
                        'value': [l1_norm(model.beta()), l2_norm(model.beta())],
                        'epoch': [i, i]
                    })
                )
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold)
    with torch.no_grad():
        new_df = pd.DataFrame({
            'loss': [F.mse_loss(model(val_x), val_y).item()]
        })
        new_df['epoch'] = i
        test_preds.append(new_df)
        norms.append(
            pd.DataFrame({
                'norm': ['l1', 'l2'],
                'value': [l1_norm(model.beta()), l2_norm(model.beta())],
                'epoch': [i, i]
            })
        )
    losses = pd.DataFrame({
        'epoch': np.arange(len(losses)),
        'loss': torch.stack(losses).numpy()
    })
    losses['split'] = 'train'
    test_preds = pd.concat(test_preds).reset_index(drop=True)
    test_preds['split'] = 'val'
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), model, pd.concat(norms).reset_index()

def teacher(x, W, V):
    outp = x@W.T
    outp = V*outp
    return outp.sum(dim=-1)

def circular_sample(shape):
    W = Normal(0,1).sample(shape)
    return W/torch.sqrt(torch.mean(W**2, dim=-1, keepdims=True))

def sample_teacher(inp_dim, active_dim):
    W = F.one_hot(torch.randperm(inp_dim)[:active_dim], inp_dim).float()
    V = torch.sign(torch.rand((active_dim,))-0.5).float()/math.sqrt(active_dim)
    return (W, V)

def main(args):
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    param = sample_teacher(args.inp_dim, args.active_dim)
    x = Normal(0, 1).sample((args.n_train, args.inp_dim))
    x = x/torch.sqrt(torch.mean(x**2, dim=-1, keepdims=True))
    val_x = Normal(0, 1).sample((10000, args.inp_dim))
    val_x = val_x/torch.sqrt(torch.mean(val_x**2, dim=-1, keepdims=True))
    y = teacher(x, *param)
    val_y = teacher(val_x, *param)
    net = DiagonalNet(args.inp_dim, scaling=args.scaling)
    df, net, norm_df = train(net, (x, y), (val_x, val_y), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold)
    df.to_feather(os.path.join(args.save_folder, 'df.feather'))
    norm_df.to_feather(os.path.join(args.save_folder, 'norm_df.feather'))
    teacher_df = pd.DataFrame({
        'norm': ['l1', 'l2'],
        'value': [l1_norm(param[1]), l2_norm(param[1])]
    })
    teacher_df.to_feather(os.path.join(args.save_folder, 'teacher_df.feather'))
    torch.save(net.state_dict(), os.path.join(args.save_folder, 'model.pt'))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--n_train', type=int, default=50)
    parser.add_argument('--inp_dim', type=int, default=100)
    parser.add_argument('--active_dim', type=int, default=10)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--no_tuning', action='store_true')
    parser.add_argument('--lr', type=float, default=1e20)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--scaling', type=float, default=1.)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
