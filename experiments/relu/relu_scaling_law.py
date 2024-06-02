from copy import deepcopy
import argparse
import math
import sys
import os
from pathlib import Path
sys.path.append('')

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import ortho_group

import functions.networks as nt

def train(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    x, y = train_data
    val_x, val_y = val_data
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        loss = F.mse_loss(model(x)[:,0], y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                new_df = pd.DataFrame({
                    'loss': [F.mse_loss(model(val_x)[:,0], val_y).item()]
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
        if i % 1000 == 0:
            print(f'Loss: {loss}')
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold)
    with torch.no_grad():
        new_df = pd.DataFrame({
            'loss': [F.mse_loss(model(val_x)[:,0], val_y).item()]
        })
        new_df['epoch'] = i
        test_preds.append(new_df)
    with torch.no_grad():
        norms = pd.DataFrame({
            'norm': ['F1', 'F2'],
            'value': [f1_norm(model).item(), f2_norm(model).item()]
        })
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
    ]).reset_index(drop=True), model, norms

def relu(x, W, V):
    outp = x@W.T
    outp = torch.relu(outp)
    outp = V*outp
    return outp

def circular_sample(shape):
    W = Normal(0,1).sample(shape)
    return W/torch.sqrt(torch.mean(W**2, dim=-1, keepdims=True))

def sample_teacher(shape, orthogonal=False, seed=0):
    if orthogonal:
        W = ortho_group.rvs(shape[1], random_state=np.random.RandomState(seed=seed))[:shape[0],:shape[1]]
        W = torch.from_numpy(W).float()
    else:
        W = circular_sample(shape)
    V = torch.sign(torch.rand((shape[0],))-0.5).float()/math.sqrt(shape[0])
    return (W, V)

def f1_norm(model):
    return torch.sum(
        torch.sqrt(torch.sum(model._modules['features'][0].weight**2, dim=1))*torch.abs(model._modules['readout'].weight[0])
    )

def f2_norm(model):
    return torch.sqrt(torch.sum(
        (torch.sqrt(torch.sum(model._modules['features'][0].weight**2, dim=1))*torch.abs(model._modules['readout'].weight[0]))**2
    ))

def l1_norm(x):
    return torch.sum(torch.abs(x)).item()

def l2_norm(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2)).item()

def main(args):
    Path(args.save_folder).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    param = sample_teacher((args.n_units, args.inp_dim), orthogonal=args.orthogonal, seed=args.seed)
    x = Normal(0, 1).sample((args.n_train, args.inp_dim))
    x = x/torch.sqrt(torch.mean(x**2, dim=-1, keepdims=True))
    val_x = Normal(0, 1).sample((10000, args.inp_dim))
    val_x = val_x/torch.sqrt(torch.mean(val_x**2, dim=-1, keepdims=True))
    y = relu(x, *param).sum(dim=-1)
    val_y = relu(val_x, *param).sum(dim=-1)
    net = nt.DenseNet2(args.inp_dim, [1000], scaling=args.scaling)
    df, net, norms = train(net, (x, y), (val_x, val_y), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold)
    df.to_feather(os.path.join(args.save_folder, 'df.feather'))
    torch.save(net.state_dict(), os.path.join(args.save_folder, 'model.pt'))
    norms['type'] = 'student'
    teacher_norms = pd.DataFrame({
        'norm': ['F1', 'F2'],
        'value': [l1_norm(param[1]), l2_norm(param[1])],
        'type': 'teacher'
    })
    pd.concat([norms, teacher_norms]).reset_index(drop=True).to_feather(os.path.join(args.save_folder, 'norms_df.feather'))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_folder', type=str, required=True)
    parser.add_argument('--n_train', type=int, default=50)
    parser.add_argument('--n_units', type=int, default=6)
    parser.add_argument('--orthogonal', action='store_true')
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--no_tuning', action='store_true')
    parser.add_argument('--lr', type=float, default=1e20)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--scaling', type=float, default=1.)
    parser.add_argument('--inp_dim', type=int, default=10)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
