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
from torch.utils import data
from tqdm import tqdm
import numpy as np
import pandas as pd
from scipy.stats import ortho_group
from sklearn.linear_model import Ridge

import functions.networks as nt

def f1_norm(model):
    return (torch.sum(
        torch.sqrt(torch.sum(model._modules['features'][0].weight**2, dim=1))*torch.abs(model._modules['readout'].weight[-1])
    )).item()

def gs_norm(model):
    return (torch.sum(
        torch.sqrt(torch.sum(model._modules['features'][0].weight**2, dim=1))*\
        torch.sqrt(torch.sum(model._modules['readout'].weight**2, dim=0))
    )).item()

def train_two_tasks(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0.,
                    lr_tuning=True, test_at_end_only=False, threshold=1e-5, batch_size=128,
                    epochs_to_track=None, true_units_1=None, true_units_2=None):
    epochs_to_track = epochs_to_track or []
    w_0 = model.features[0].weight.clone().detach().numpy()
    n_hidden = w_0.shape[0]
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    df_statistics = []
    train_set = data.TensorDataset(*train_data)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=False)
    with torch.no_grad():
        t1_weight = torch.mean(1-train_data[2].float())
        t2_weight = torch.mean(train_data[2].float())
    val_x, val_y1, val_y2 = val_data
    for i in tqdm(range(epochs)):
        if i in epochs_to_track:
            with torch.no_grad():
                w = model.features[0].weight.clone().detach().numpy()
                alignment_with_units_1 = np.corrcoef(w, true_units_1)[:n_hidden, n_hidden:]
                alignment_with_units_2 = np.corrcoef(w, true_units_2)[:n_hidden, n_hidden:]
                alignment_with_original_weights = np.corrcoef(w, w_0)[:n_hidden, n_hidden:].diagonal()
                unit_magnitude = np.sqrt(np.sum(w**2, axis=1))
                readout = np.abs(model.readout.weight.detach().numpy())
                new_df = [
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': alignment_with_units_1[:,i],
                        'type': f'alignment 1 {i}'
                    }) for i in range(alignment_with_units_1.shape[1])
                ] + [
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': alignment_with_units_2[:,i],
                        'type': f'alignment 2 {i}'
                    }) for i in range(alignment_with_units_2.shape[1])
                ] + [
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': alignment_with_original_weights,
                        'type': f'alignment original'
                    }),
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': unit_magnitude,
                        'type': f'unit magnitude'
                    }),
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': readout[0],
                        'type': f'readout 1'
                    }),
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': readout[1],
                        'type': f'readout 2'
                    })
                ]
                new_df = pd.concat(new_df)
                new_df['epoch'] = i
                df_statistics.append(new_df)
        _losses = []
        for x, y, task in train_loader:
            optimizer.zero_grad()
            pred = model(x)
            pred = select_output(pred, task)
            t1_loss = F.mse_loss(pred[task==0], y[task==0])
            t1_loss = torch.where(torch.isnan(t1_loss), 0., t1_loss)
            t2_loss = F.mse_loss(pred[task==1], y[task==1])
            t2_loss = torch.where(torch.isnan(t2_loss), 0., t2_loss)
            loss = t1_weight*t1_loss+t2_weight*t2_loss
            loss.backward()
            optimizer.step()
            _losses.append(loss.item())
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
        losses.append(torch.tensor(_losses).mean().detach())
        loss = torch.tensor(_losses).mean().detach().item()
        if loss < threshold:
            break
        if lr_tuning and ((loss > 100) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train_two_tasks(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, batch_size=batch_size,
                                   epochs_to_track=epochs_to_track, true_units_1=true_units_1, true_units_2=true_units_2)
    with torch.no_grad():
        w = model.features[0].weight.clone().detach().numpy()
        alignment_with_units_1 = np.corrcoef(w, true_units_1)[:n_hidden, n_hidden:]
        alignment_with_units_2 = np.corrcoef(w, true_units_2)[:n_hidden, n_hidden:]
        alignment_with_original_weights = np.corrcoef(w, w_0)[:n_hidden, n_hidden:].diagonal()
        unit_magnitude = np.sqrt(np.sum(w**2, axis=1))
        readout = np.abs(model.readout.weight.detach().numpy())
        new_df = [
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': alignment_with_units_1[:,i],
                'type': f'alignment 1 {i}'
            }) for i in range(alignment_with_units_1.shape[1])
        ] + [
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': alignment_with_units_2[:,i],
                'type': f'alignment 2 {i}'
            }) for i in range(alignment_with_units_2.shape[1])
        ] + [
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': alignment_with_original_weights,
                'type': f'alignment original'
            }),
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': unit_magnitude,
                'type': f'unit magnitude'
            }),
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': readout[0],
                'type': f'readout 1'
            }),
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': readout[1],
                'type': f'readout 2'
            })
        ]
        new_df = pd.concat(new_df)
        new_df['epoch'] = i
        df_statistics.append(new_df)
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
    df_statistics = pd.concat(df_statistics).reset_index(drop=True)
    df_norm = pd.DataFrame({
        'norm': ['F1', 'GS'],
        'value': [f1_norm(model), gs_norm(model)],
        'kind': 'student'
    })
    return pd.concat([
        losses,
        test_preds
    ]).reset_index(drop=True), df_statistics, df_norm

def train_one_task(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True,
          test_at_end_only=False, threshold=1e-5, batch_size=128, epochs_to_track=None, true_units_1=None, true_units_2=None):
    epochs_to_track = epochs_to_track or []
    w_0 = model.features[0].weight.clone().detach().numpy()
    n_hidden = w_0.shape[0]
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    df_statistics = []
    train_set = data.TensorDataset(*train_data)
    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_x, val_y = val_data
    for i in tqdm(range(epochs)):
        if i in epochs_to_track:
            with torch.no_grad():
                w = model.features[0].weight.clone().detach().numpy()
                alignment_with_units_1 = np.corrcoef(w, true_units_1)[:n_hidden, n_hidden:]
                alignment_with_units_2 = np.corrcoef(w, true_units_2)[:n_hidden, n_hidden:]
                alignment_with_original_weights = np.corrcoef(w, w_0)[:n_hidden, n_hidden:].diagonal()
                unit_magnitude = np.sqrt(np.sum(w**2, axis=1))
                readout_magnitude = np.abs(model.readout.weight[0].detach().numpy())
                new_df = [
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': alignment_with_units_1[:,i],
                        'type': f'alignment 1 {i}'
                    }) for i in range(alignment_with_units_1.shape[1])
                ] + [
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': alignment_with_units_2[:,i],
                        'type': f'alignment 2 {i}'
                    }) for i in range(alignment_with_units_2.shape[1])
                ] + [
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': alignment_with_original_weights,
                        'type': f'alignment original'
                    }),
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': unit_magnitude,
                        'type': f'unit magnitude'
                    }),
                    pd.DataFrame({
                        'unit': list(range(n_hidden)),
                        'value': readout_magnitude,
                        'type': f'readout'
                    })
                ]
                new_df = pd.concat(new_df)
                new_df['epoch'] = i
                df_statistics.append(new_df)
        _losses = []
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = F.mse_loss(model(x)[:,0], y)
            loss.backward()
            optimizer.step()
            _losses.append(loss.detach().item())
            if lr_tuning and ((loss.detach().numpy() > 1000) | np.isnan(loss.detach().numpy())):
                lr = lr/10
                print(f'Decreasing learning rate to {lr}')
                return train_one_task(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold, batch_size=batch_size)
        loss = torch.tensor(_losses).mean().detach()
        losses.append(loss)
        if loss.item() < threshold:
            break
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                new_df = pd.DataFrame({
                    'loss': [F.mse_loss(model(val_x)[:,0], val_y).item()]
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
    with torch.no_grad():
        w = model.features[0].weight.clone().detach().numpy()
        alignment_with_units_1 = np.corrcoef(w, true_units_1)[:n_hidden, n_hidden:]
        alignment_with_units_2 = np.corrcoef(w, true_units_2)[:n_hidden, n_hidden:]
        alignment_with_original_weights = np.corrcoef(w, w_0)[:n_hidden, n_hidden:].diagonal()
        unit_magnitude = np.sqrt(np.sum(w**2, axis=1))
        readout_magnitude = np.abs(model.readout.weight[0].detach().numpy())
        new_df = [
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': alignment_with_units_1[:,i],
                'type': f'alignment 1 {i}'
            }) for i in range(alignment_with_units_1.shape[1])
        ] + [
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': alignment_with_units_2[:,i],
                'type': f'alignment 2 {i}'
            }) for i in range(alignment_with_units_2.shape[1])
        ] + [
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': alignment_with_original_weights,
                'type': f'alignment original'
            }),
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': unit_magnitude,
                'type': f'unit magnitude'
            }),
            pd.DataFrame({
                'unit': list(range(n_hidden)),
                'value': readout_magnitude,
                'type': f'readout magnitude'
            })
        ]
        new_df = pd.concat(new_df)
        new_df['epoch'] = i
        df_statistics.append(new_df)
    with torch.no_grad():
        new_df = pd.DataFrame({
            'loss': [F.mse_loss(model(val_x)[:,0], val_y).item()]
        })
        new_df['epoch'] = i
        test_preds.append(new_df)
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
    ]).reset_index(drop=True), pd.concat(df_statistics).reset_index(drop=True), f1_norm(model)

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

def sample_two_teachers(inp_dim, n_units_1, n_units_2, seed, correlation=[1.]*6):
    W = ortho_group.rvs(inp_dim, random_state=np.random.RandomState(seed=seed))
    W = torch.from_numpy(W).float()
    V = (torch.rand(n_units_1)-0.5).sign()/math.sqrt(n_units_1)
    a = torch.tensor(correlation)
    b = torch.sqrt(1-a**2)
    W2 = a.unsqueeze(1)*W[:n_units_2]+b.unsqueeze(1)*W[n_units_1:(n_units_1+n_units_2)]
    W = W[:n_units_1]
    V2 = (torch.rand(n_units_2)-0.5).sign()/math.sqrt(n_units_2)
    return (W, V), (W2, V2)

def select_output(outp, task):
    task_oh = F.one_hot(task, outp.shape[1])
    return (outp*task_oh).sum(dim=-1)

def main(args):
    Path(os.path.dirname(args.save_path)).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    param1, param2 = sample_two_teachers(args.inp_dim, args.n_units_1, args.n_units_2, args.seed, correlation=args.correlation)
    x1 = circular_sample(args.n_train1, args.inp_dim)
    x2 = circular_sample(args.n_train2, args.inp_dim, variance=args.train_var)
    val_x = circular_sample(10000, args.inp_dim)
    y1 = relu(x1, *param1).sum(dim=-1)
    y2 = relu(x2, *param2).sum(dim=-1)
    x = torch.cat([x1, x2])
    y = torch.cat([y1, y2])
    task = torch.tensor([0]*args.n_train1+[1]*args.n_train2)
    val_y1 = relu(val_x, *param1).sum(dim=-1)
    val_y2 = relu(val_x, *param2).sum(dim=-1)
    if args.load_model:
        net = nt.DenseNet2(args.inp_dim, [args.hdims], outp_dim=1, scaling=args.model_scaling)
        net.load_state_dict(torch.load(args.model_path))
        nn.init.normal_(net.readout.weight, std=args.readout_scaling*math.sqrt(2/net.readout.weight.shape[1]))
        net.features[0].weight = nn.Parameter(args.weight_scaling*net.features[0].weight)
    else:
        outp_dim = 1 if args.one_task else 2
        net = nt.DenseNet2(args.inp_dim, [args.hdims], outp_dim=outp_dim, scaling=args.scaling)
    if args.one_task:
        if args.setup in ['linear_readout', 'ntk']:
            if args.setup == 'ntk':
                h = net.ntk_features(x2)
                val_h = net.ntk_features(val_x)
            with torch.no_grad():
                if args.setup == 'linear_readout':
                    h = net.features(x2)
                    val_h = net.features(val_x)
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
            if args.setup == 'lp_then_bp':
                net.linear_readout = True
                df_lp, df_statistics_lp, _ = train_one_task(
                    net, (x2, y2), (val_x, val_y2), lr=args.lr, epochs=args.epochs,
                    lr_tuning=(not args.no_tuning), threshold=args.lp_threshold,
                    epochs_to_track=args.epochs_to_track, true_units_1=param1[0].detach(),
                    true_units_2=param2[0].detach())
                net.linear_readout = False
            df, df_statistics, norm = train_one_task(
                net, (x2, y2), (val_x, val_y2), lr=args.lr, epochs=args.epochs,
                lr_tuning=(not args.no_tuning), threshold=args.threshold,
                epochs_to_track=args.epochs_to_track, true_units_1=param1[0].detach(),
                true_units_2=param2[0].detach())
            with torch.no_grad():
                net = nt.DenseNet(args.inp_dim, [1000], outp_dim=1)
                net.load_state_dict(torch.load(args.model_path))
                gs_norm = norm + f1_norm(net)
            n_equal = torch.sum((torch.tensor(args.correlation)==1.).float()).item()
            df_norm = pd.DataFrame({
                'norm': ['F1', 'GS', 'F1', 'GS'],
                'kind': ['student', 'student', 'teacher', 'teacher'],
                'value': [norm, gs_norm, math.sqrt(args.n_units_2),
                          (args.n_units_1-n_equal)/(math.sqrt(args.n_units_1))+(args.n_units_2-n_equal)/(math.sqrt(args.n_units_2))+n_equal*math.sqrt(1/args.n_units_1+1/args.n_units_2)]
            })
            df_norm.to_feather(os.path.join(args.save_path, 'df_norm.feather'))
            if args.setup == 'lp_then_bp':
                df_lp['stage'] = 'lp'
                df['stage'] = 'bp'
                df = pd.concat([df_lp, df]).reset_index(drop=True)
                df_statistics_lp['stage'] = 'lp'
                df_statistics['stage'] = 'bp'
                df_statistics = pd.concat([df_statistics_lp, df_statistics]).reset_index(drop=True)
            df_statistics.to_feather(os.path.join(args.save_path, 'df_statistics.feather'))
    else:
        df, df_statistics, df_norm = train_two_tasks(net, (x, y, task), (val_x, val_y1, val_y2), lr=args.lr, epochs=args.epochs, lr_tuning=(not args.no_tuning), threshold=args.threshold,
                                      epochs_to_track=args.epochs_to_track, true_units_1=param1[0].detach(), true_units_2=param2[0].detach())
        df_statistics.to_feather(os.path.join(args.save_path, 'df_statistics.feather'))
        df_norm.to_feather(os.path.join(args.save_path, 'df_norm.feather'))
        with torch.no_grad():
            n_equal = torch.sum((torch.tensor(args.correlation)==1.).float()).item()
            l1_norm = math.sqrt(args.n_units_2)
            gs_norm = (args.n_units_1-n_equal)/(math.sqrt(args.n_units_1))+(args.n_units_2-n_equal)/(math.sqrt(args.n_units_2))+n_equal*math.sqrt(1/args.n_units_1+1/args.n_units_2)
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
    parser.add_argument('--n_units_1', type=int, default=6)
    parser.add_argument('--n_units_2', type=int, default=6)
    parser.add_argument('--threshold', type=float, default=1e-6)
    parser.add_argument('--no_tuning', action='store_true')
    parser.add_argument('--lr', type=float, default=1e20)
    parser.add_argument('--epochs', type=int, default=int(1e5))
    parser.add_argument('--scaling', type=float, default=1.)
    parser.add_argument('--correlation', type=float, default=[1.]*6, nargs='+')
    parser.add_argument('--train_var', type=float, default=[1.]*15, nargs='+')
    parser.add_argument('--one_task', action='store_true')
    parser.add_argument('--setup', choices=['backprop', 'ntk', 'linear_readout', 'lp_then_bp'], default='backprop')
    parser.add_argument('--inp_dim', default=15, type=int)
    parser.add_argument('--epochs_to_track', default=None, type=int, nargs='+')
    parser.add_argument('--hdims', type=int, default=1000)
    parser.add_argument('--readout_scaling', type=float, default=1e-3)
    parser.add_argument('--weight_scaling', type=float, default=1.)
    parser.add_argument('--lp_threshold', type=float, default=0.1)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
