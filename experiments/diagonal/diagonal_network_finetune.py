from copy import deepcopy
import argparse
import sys
import math
import os
from pathlib import Path
sys.path.append('')

import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions.normal import Normal
from tqdm import tqdm
import numpy as np
import pandas as pd

import functions.networks as nt

class DiagonalNet(nn.Module):
    def __init__(self, inp_dim, scaling=1., linear_readout=False):
        super().__init__()
        self.w_pos = nn.Parameter(scaling*torch.ones(inp_dim))
        self.v_pos = nn.Parameter(scaling*torch.ones(inp_dim))
        self.v_neg = nn.Parameter(scaling*torch.ones(inp_dim))
        self.w_neg = nn.Parameter(scaling*torch.ones(inp_dim))
        self.linear_readout = linear_readout
    
    def beta(self):
        return self.w_pos*self.v_pos-self.w_neg*self.v_neg

    def parameters(self):
        if self.linear_readout:
            return [self.v_pos, self.v_neg]
        else:
            return [self.w_pos, self.v_pos, self.v_neg, self.w_neg]

    def forward(self, x):
        return x@self.beta()

def l1_norm(x):
    return torch.sum(torch.abs(x)).item()

def l2_norm(x):
    return torch.sqrt(torch.sum(torch.abs(x)**2)).item()

def mt_norm(x):
    piecewise_l2 = torch.sqrt(torch.sum(x**2, dim=1))
    return piecewise_l2.sum().item()

def q(z):
    return 2-torch.sqrt(4+z**2)+z*torch.arcsinh(z/2)

def q_norm(x, gamma):
    return ((torch.abs(x[:,0])+gamma**2)*q(x[:,1]/(torch.abs(x[:,0])+gamma**2))).sum().item()

class MTDiagonalNet(nn.Module):
    def __init__(self, inp_dim, outp_dim=1, scaling=1., linear_readout=False):
        super().__init__()
        self.w_pos = nn.Parameter(scaling*torch.ones(inp_dim, 1))
        self.v_pos = nn.Parameter(scaling*torch.ones(inp_dim, outp_dim))
        self.v_neg = nn.Parameter(scaling*torch.ones(inp_dim, outp_dim))
        self.w_neg = nn.Parameter(scaling*torch.ones(inp_dim, 1))
        self.linear_readout = linear_readout
    
    def beta(self):
        return self.w_pos*self.v_pos-self.w_neg*self.v_neg

    def parameters(self):
        if self.linear_readout:
            return [self.v_pos, self.v_neg]
        else:
            return [self.w_pos, self.v_pos, self.v_neg, self.w_neg]

    def forward(self, x):
        return x@self.beta()

def train_two_tasks(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5, pretrained_beta=None):
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
                val_mse_1 = F.mse_loss(model(val_x)[:,0], val_y1).item()
                val_mse_2 = F.mse_loss(model(val_x)[:,1], val_y2).item()
                print(f'Epoch {i}: Validation MSE Task 1: {val_mse_1:.6f}, Validation MSE Task 2: {val_mse_2:.6f}')
                new_df = pd.DataFrame({
                    'loss': [val_mse_1, val_mse_2],
                    'split': ['val_1', 'val_2']
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train_two_tasks(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, pretrained_beta=pretrained_beta)
    with torch.no_grad():
        final_val_mse_1 = F.mse_loss(model(val_x)[:,0], val_y1).item()
        final_val_mse_2 = F.mse_loss(model(val_x)[:,1], val_y2).item()
        print(f'Final Validation MSE Task 1: {final_val_mse_1:.6f}, Final Validation MSE Task 2: {final_val_mse_2:.6f}')
        new_df = pd.DataFrame({
            'loss': [final_val_mse_1, final_val_mse_2],
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
    with torch.no_grad():
        both_betas = model.beta()
        q_betas = torch.stack([pretrained_beta, both_betas[:,1]], dim=1)
        norm_df = pd.DataFrame({
            'norm': ['l1', 'l2', 'mt', 'q'],
            'value': [l1_norm(both_betas[:,1]), l2_norm(both_betas[:,1]), mt_norm(both_betas), q_norm(q_betas, args.scaling)],
            'kind': 'student'
        })
        df_weights = pd.concat([
            pd.DataFrame({'dim': np.arange(both_betas.shape[0]), 'value': both_betas[:,0], 'task': '1'}),
            pd.DataFrame({'dim': np.arange(both_betas.shape[0]), 'value': both_betas[:,1], 'task': '2'})
        ])
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), norm_df, model, df_weights.reset_index(drop=True)

def train_one_task(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5, beta_1=None):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    x, y = train_data
    val_x, val_y = val_data
    for i in tqdm(range(epochs)):
        optimizer.zero_grad()
        pred = model(x)
        loss = F.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                val_mse = F.mse_loss(model(val_x), val_y).item()
                print(f'Epoch {i}: Validation MSE: {val_mse:.6f}')
                new_df = pd.DataFrame({
                    'loss': [val_mse],
                    'split': ['val']
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train_one_task(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, beta_1=beta_1)
    with torch.no_grad():
        final_val_mse = F.mse_loss(model(val_x), val_y).item()
        print(f'Final Validation MSE: {final_val_mse:.6f}')
        new_df = pd.DataFrame({
            'loss': [final_val_mse],
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
    with torch.no_grad():
        beta = model.beta()
        both_betas = torch.stack([beta_1, beta], dim=1)
        norm_df = pd.DataFrame({
            'norm': ['l1', 'l2', 'mt', 'q'],
            'value': [l1_norm(beta), l2_norm(beta), mt_norm(both_betas), q_norm(both_betas, args.scaling)],
            'kind': 'student'
        })
        df_weights = pd.concat([
            pd.DataFrame({'dim': np.arange(both_betas.shape[0]), 'value': both_betas[:,0], 'task': '1'}),
            pd.DataFrame({'dim': np.arange(both_betas.shape[0]), 'value': both_betas[:,1], 'task': '2'})
        ])
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), norm_df, model, df_weights.reset_index(drop=True)

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

def sample_two_teachers(inp_dim, active_dim_1, active_dim_2, overlap=0):
    perm = torch.randperm(inp_dim)
    W = F.one_hot(perm[:active_dim_1], inp_dim).float()
    V = torch.sign(torch.rand((active_dim_1,))-0.5).float()
    W2 = F.one_hot(
        torch.cat([perm[:overlap], perm[active_dim_1:(active_dim_1+active_dim_2-overlap)]]),
        inp_dim
    ).float()
    return (W, V/math.sqrt(active_dim_1)), (W2, V[:active_dim_2]/math.sqrt(active_dim_2))

def select_output(outp, task):
    task_oh = F.one_hot(task, outp.shape[1])
    return (outp*task_oh).sum(dim=-1)

def main(args):
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    # Print experimental setup
    setup = {
        'seed': args.seed,
        'save_path': args.save_path,
        'model_path': args.model_path,
        'load_model': bool(args.load_model),
        'model_scaling': args.model_scaling,
        'n_train1': args.n_train1,
        'n_train2': args.n_train2,
        'active_dim_1': args.active_dim_1,
        'active_dim_2': args.active_dim_2,
        'inp_dim': args.inp_dim,
        'threshold': args.threshold,
        'no_tuning': bool(args.no_tuning),
        'lr': args.lr,
        'epochs': args.epochs,
        'scaling': args.scaling,
        'overlap': args.overlap,
        'linear_readout': bool(args.linear_readout),
        'one_task': bool(args.one_task),
        'save_weights': bool(args.save_weights),
        'w_scaling': args.w_scaling,
        'init_method': getattr(args, 'init_method', None),
        'lmda': getattr(args, 'lmda', None),
        'c': getattr(args, 'c', None),
    }
    print('Experimental setup:')
    for k, v in setup.items():
        print(f'  {k}: {v}')
    param1, param2 = sample_two_teachers(args.inp_dim, args.active_dim_1, args.active_dim_2, overlap=args.overlap)
    x1 = circular_sample((args.n_train1, args.inp_dim))
    x2 = circular_sample((args.n_train2, args.inp_dim))
    val_x = circular_sample((10000, args.inp_dim))
    y1 = teacher(x1, *param1)
    y2 = teacher(x2, *param2)
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    task = torch.tensor([0]*args.n_train1+[1]*args.n_train2)
    val_y1 = teacher(val_x, *param1)
    val_y2 = teacher(val_x, *param2)
    net = DiagonalNet(args.inp_dim, scaling=args.model_scaling, linear_readout=args.linear_readout)
    net.load_state_dict(torch.load(args.model_path))
    pretrained_beta = net.beta().detach().clone()
    if not args.load_model:
        if args.one_task:
            net = DiagonalNet(args.inp_dim, scaling=args.scaling, linear_readout=args.linear_readout)
        else:
            net = MTDiagonalNet(args.inp_dim, outp_dim=2, scaling=args.scaling, linear_readout=args.linear_readout)
    if args.one_task:
        nn.init.constant_(net.v_pos, args.scaling)
        nn.init.constant_(net.v_neg, args.scaling)
        net.w_pos = nn.Parameter(args.w_scaling*net.w_pos)
        net.w_neg = nn.Parameter(args.w_scaling*net.w_neg)
        df, norm_df, model, df_weights = train_one_task(net, (x2, y2), (val_x, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold, beta_1=pretrained_beta)
    else:
        df, norm_df, model, df_weights = train_two_tasks(net, (x, y, task), (val_x, val_y1, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold, pretrained_beta=pretrained_beta)
    true_beta = torch.stack([
        param1[0].T@(param1[1]),
        param2[0].T@(param2[1])
    ], dim=1)
    norm_df = pd.concat([
        norm_df,
        pd.DataFrame({
            'norm': ['l1', 'l2', 'mt', 'q'],
            'value': [l1_norm(true_beta), l2_norm(true_beta), mt_norm(true_beta), q_norm(torch.stack([pretrained_beta, true_beta[:,1]], dim=1), args.scaling)],
            'kind': 'teacher'
        })
    ]).reset_index(drop=True)
    df.to_feather(os.path.join(args.save_path, 'df.feather'))
    norm_df.to_feather(os.path.join(args.save_path, 'norm_df.feather'))
    if args.save_weights:
        print('Saving weights')
        print(os.path.join(args.save_path, 'weights_df.feather'))
        df_weights.to_feather(os.path.join(args.save_path, 'weights_df.feather')) 

    # Append summary row with final metrics and experimental setup
    with torch.no_grad():
        if args.one_task:
            final_train_mse = F.mse_loss(model(x2), y2).item()
            final_val_mse = F.mse_loss(model(val_x), val_y2).item()
            metrics = {
                'final_train_mse': final_train_mse,
                'final_val_mse': final_val_mse,
                'final_train_mse_task1': np.nan,
                'final_train_mse_task2': np.nan,
                'final_val_mse_task1': np.nan,
                'final_val_mse_task2': np.nan,
            }
        else:
            pred_train = model(x)
            final_train_mse_task1 = F.mse_loss(pred_train[task==0, 0], y[task==0]).item()
            final_train_mse_task2 = F.mse_loss(pred_train[task==1, 1], y[task==1]).item()
            final_val_mse_task1 = F.mse_loss(model(val_x)[:,0], val_y1).item()
            final_val_mse_task2 = F.mse_loss(model(val_x)[:,1], val_y2).item()
            metrics = {
                'final_train_mse': np.nan,
                'final_val_mse': np.nan,
                'final_train_mse_task1': final_train_mse_task1,
                'final_train_mse_task2': final_train_mse_task2,
                'final_val_mse_task1': final_val_mse_task1,
                'final_val_mse_task2': final_val_mse_task2,
            }

    # Collect experimental setup variables
    exp_vars = {
        'seed': args.seed,
        'save_path': args.save_path,
        'model_path': args.model_path,
        'load_model': bool(args.load_model),
        'model_scaling': args.model_scaling,
        'n_train1': args.n_train1,
        'n_train2': args.n_train2,
        'active_dim_1': args.active_dim_1,
        'active_dim_2': args.active_dim_2,
        'inp_dim': args.inp_dim,
        'threshold': args.threshold,
        'no_tuning': bool(args.no_tuning),
        'lr': args.lr,
        'epochs': args.epochs,
        'scaling': args.scaling,
        'overlap': args.overlap,
        'linear_readout': bool(args.linear_readout),
        'one_task': bool(args.one_task),
        'save_weights': bool(args.save_weights),
        'w_scaling': args.w_scaling,
        # Pretrain hyperparameters passed through for traceability
        'init_method': getattr(args, 'init_method', None),
        'lmda': getattr(args, 'lmda', None),
        'c': getattr(args, 'c', None),
    }
    row = {**metrics, **exp_vars}
    # Write to central CSV at the suite root (parent of the experiment folder)
    results_path = os.path.join(os.path.dirname(os.path.dirname(args.save_path)), 'experiment_results.csv')
    try:
        if os.path.exists(results_path):
            existing = pd.read_csv(results_path)
            new_df = pd.concat([existing, pd.DataFrame([row])], ignore_index=True)
        else:
            new_df = pd.DataFrame([row])
        new_df.to_csv(results_path, index=False)
    except Exception as e:
        print(f'Warning: failed to write experiment_results.csv due to: {e}')

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--model_scaling', type=float, default=1.)
    parser.add_argument('--n_train1', type=int, default=50)
    parser.add_argument('--n_train2', type=int, default=50)
    parser.add_argument('--active_dim_1', type=int, default=10)
    parser.add_argument('--active_dim_2', type=int, default=10)
    parser.add_argument('--inp_dim', type=int, default=1000)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--no_tuning', action='store_true')
    parser.add_argument('--lr', type=float, default=1e20)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--scaling', type=float, default=1.)
    parser.add_argument('--overlap', type=int, default=0)
    parser.add_argument('--linear_readout', action='store_true')
    parser.add_argument('--one_task', action='store_true')
    parser.add_argument('--save_weights', action='store_true')
    parser.add_argument('--w_scaling', type=float, default=1.)
    # Accept but ignore these, so array drivers can pass pretrain hyperparams
    parser.add_argument('--init_method', type=str, choices=['complex', 'simple'], default='complex')
    parser.add_argument('--lmda', type=str, default='0.0000000000')
    parser.add_argument('--c', type=str, default='0.0010000000')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
