import argparse

import torch
from numpy.random import RandomState
from torchvision.datasets import MNIST as OriginalMNIST
from torchvision import transforms
import torch.nn as nn
from torch.utils import data

class MNIST:
    def __init__(self, hparams=None, **kwargs):
        super().__init__()
        if hparams is None:
            parser = argparse.ArgumentParser()
            parser = MNIST.add_args(parser)
            hparams = parser.parse_args([])
        if isinstance(hparams, dict):
            hparams = argparse.Namespace(**hparams)
        for key, value in kwargs.items():
            hparams.__setattr__(key, value)
        self.hparams = hparams
        rs = RandomState(seed=hparams.split_seed)
        mnist = OriginalMNIST(
            root=hparams.root,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(0.5, 0.5),
                nn.Flatten(start_dim=0)
            ])
        )
        cat_indices = {i: [] for i in range(10)}
        for j, (_, y) in enumerate(mnist):
            cat_indices[y].append(j)
        for i in range(10):
            cat_indices[i] = rs.permutation(cat_indices[i])
        train_x, train_y, train_task = [], [], []
        val_data = []
        for task, (cat_1, cat_2) in enumerate(hparams.cats):
            for index in cat_indices[cat_1][:hparams.n_train]:
                x, y = mnist[index]
                train_x.append(x)
                train_y.append(1.)
                train_task.append(task)
            for index in cat_indices[cat_2][:hparams.n_train]:
                x, y = mnist[index]
                train_x.append(x)
                train_y.append(-1.)
                train_task.append(task)
            new_val_x, new_val_y, new_val_task = [], [], []
            for index in cat_indices[cat_1][(len(cat_indices[cat_1])-hparams.n_val):]:
                x, _ = mnist[index]
                new_val_x.append(x)
                new_val_y.append(1.)
                new_val_task.append(task)
            for index in cat_indices[cat_2][(len(cat_indices[cat_2])-hparams.n_val):]:
                x, _ = mnist[index]
                new_val_x.append(x)
                new_val_y.append(-1.)
                new_val_task.append(task)
            new_val_x = torch.stack(new_val_x, dim=0)
            new_val_y = torch.stack(new_val_y, dim=0)
            new_val_task = torch.stack(new_val_task, dim=0)
            val_data.append(
                data.TensorDataset(new_val_x, new_val_y, new_val_task)
            )
        
        cat_indices = {
            i: rs.permutation([j for j,(_,y) in enumerate(mnist) if y == i])\
            for i in range(hparams.n_cats)
        }
        train_indices = {
            k: v[:hparams.n_train] for k,v in cat_indices.items()
        }
        val_indices = {
            k: v[(len(v)-hparams.n_val):] for k,v in cat_indices.items()
        }
        self.train_set = CategoricalTI(
            dataset=mnist,
            indices=train_indices,
            random_state=RandomState(seed=hparams.train_seed),
            mode='train'
        )
        self.val_sets = [
            CategoricalTI(
                dataset=mnist,
                indices=train_indices,
                random_state=RandomState(seed=hparams.train_seed),
                mode='test'
            ),
            CategoricalTI(
                dataset=mnist,
                indices=val_indices,
                random_state=RandomState(seed=hparams.val_seed),
                mode='test'
            )
        ]
    
    def get_train_loader(self):
        train_loader = data.DataLoader(
            self.train_set, batch_size=self.hparams.batch_size, shuffle=True, num_workers=8
        )
        return train_loader

    def get_val_loaders(self):
        val_loaders = [
            data.DataLoader(
                val_set, batch_size=self.hparams.batch_size, shuffle=False, num_workers=8
            )\
            for val_set in self.val_sets
        ]
        return val_loaders

    @staticmethod
    def add_args(parser):
        parser.add_argument('--split_seed', type=int, default=0)
        parser.add_argument('--train_seed', type=int, default=0)
        parser.add_argument('--val_seed', type=int, default=0)
        parser.add_argument('--n_train', type=int, default=1000)
        parser.add_argument('--n_val', type=int, default=1000)
        parser.add_argument('--n_cats', type=int, default=7)
        parser.add_argument('--batch_size', type=int, default=128)
        parser.add_argument('--root', type=str, default='./_data')
        return parser
