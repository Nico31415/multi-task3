import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler

import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.utils.data as data
import pandas as pd
from tqdm import tqdm

import argparse
import pathlib
import os
import time
import copy

class ViT(nn.Module):
    def __init__(self, in_c:int=3, num_classes:int=10, img_size:int=32, patch:int=8, dropout:float=0., num_layers:int=7, hidden:int=384, mlp_hidden:int=384*4, head:int=8, is_cls_token:bool=True):
        super(ViT, self).__init__()
        # hidden=384

        self.patch = patch # number of patches in one row(or col)
        self.is_cls_token = is_cls_token
        self.patch_size = img_size//self.patch
        f = (img_size//self.patch)**2*3 # 48 # patch vec length
        num_tokens = (self.patch**2)+1 if self.is_cls_token else (self.patch**2)

        self.emb = nn.Linear(f, hidden) # (b, n, f)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden)) if is_cls_token else None
        self.pos_emb = nn.Parameter(torch.randn(1,num_tokens, hidden))
        enc_list = [TransformerEncoder(hidden,mlp_hidden=mlp_hidden, dropout=dropout, head=head) for _ in range(num_layers)]
        self.enc = nn.Sequential(*enc_list)
        self.fc = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, num_classes) # for cls_token
        )

    def forward(self, x):
        out = self._to_words(x)
        out = self.emb(out)
        if self.is_cls_token:
            out = torch.cat([self.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + self.pos_emb
        out = self.enc(out)
        if self.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = self.fc(out)
        return out

    def _to_words(self, x):
        """
        (b, c, h, w) -> (b, n, f)
        """
        out = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size).permute(0,2,3,4,5,1)
        out = out.reshape(x.size(0), self.patch**2 ,-1)
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, feats:int, mlp_hidden:int, head:int=8, dropout:float=0.):
        super(TransformerEncoder, self).__init__()
        self.la1 = nn.LayerNorm(feats)
        self.msa = MultiHeadSelfAttention(feats, head=head, dropout=dropout)
        self.la2 = nn.LayerNorm(feats)
        self.mlp = nn.Sequential(
            nn.Linear(feats, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, feats),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out = self.msa(self.la1(x)) + x
        out = self.mlp(self.la2(out)) + out
        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, feats:int, head:int=8, dropout:float=0.):
        super(MultiHeadSelfAttention, self).__init__()
        self.head = head
        self.feats = feats
        self.sqrt_d = self.feats**0.5

        self.q = nn.Linear(feats, feats)
        self.k = nn.Linear(feats, feats)
        self.v = nn.Linear(feats, feats)

        self.o = nn.Linear(feats, feats)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, f = x.size()
        q = self.q(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        k = self.k(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)
        v = self.v(x).view(b, n, self.head, self.feats//self.head).transpose(1,2)

        score = F.softmax(torch.einsum("bhif, bhjf->bhij", q, k)/self.sqrt_d, dim=-1) #(b,h,n,n)
        attn = torch.einsum("bhij, bhjf->bihf", score, v) #(b,n,h,f//h)
        o = self.dropout(self.o(attn.flatten(2)))
        return o

def data_to_numpy(train_set, test_set, train_indices, test_indices):
    batch_size=max(len(train_indices), len(test_indices))
    dataloaders, dataset_sizes = build_dataloaders(train_set, test_set, train_indices, test_indices, batch_size)

    for inputs, labels in dataloaders['train']:
        X_train = inputs.data.numpy()
        y_train = labels.data.numpy()
        
    for inputs, labels in dataloaders['valid']:
        X_valid = inputs.data.numpy()
        y_valid = labels.data.numpy()

    return X_train, y_train, X_valid, y_valid

def build_dataloaders(train_set, test_set, train_indices, test_indices, batch_size=128):
    """ Returns dataloaders and dataset_sizes """
    dataloaders = {
        'train': torch.utils.data.DataLoader(
            train_set, batch_size=batch_size, 
            sampler=SubsetRandomSampler(train_indices)
        ),
        'valid': torch.utils.data.DataLoader(
            test_set, batch_size=batch_size, 
            sampler=SubsetRandomSampler(test_indices)
        )
    }
    dataset_sizes = {
        'train': len(train_indices),
        'valid': len(test_indices)
    }
    return dataloaders, dataset_sizes

def get_train_data(random_seed):
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = datasets.CIFAR100(root='./data', train=True,
                                           download=True, transform=transform)

    test_set = datasets.CIFAR100(root='./data', train=False,
                                           download=True, transform=transform)
    randperm = np.random.permutation(100)
    for t in range(len(train_set.targets)):
        train_set.targets[t] = randperm[train_set.targets[t]]
    for t in range(len(test_set.targets)):
        test_set.targets[t] = randperm[test_set.targets[t]]

    train_digits_04 = np.where(np.array(train_set.targets) < 2)[0]
    train_digits_59 = np.where(np.logical_and(np.array(train_set.targets) > 1, np.array(train_set.targets) > 1))[0]

    test_digits_04 = np.where(np.array(test_set.targets) < 2)[0]
    test_digits_59 = np.where(np.logical_and(np.array(test_set.targets) > 1,  np.array(test_set.targets) > 1))[0]

    (len(train_digits_04), len(test_digits_04)), (len(train_digits_59), len(test_digits_59))

    X_train_04, y_train_04, X_valid_04, y_valid_04 = data_to_numpy(train_set, test_set, train_digits_04, test_digits_04)


    X_train_59, y_train_59, X_valid_59, y_valid_59 = data_to_numpy(train_set, test_set, train_digits_59, test_digits_59)

    # fixing the issues with labels
    y_train_59 = y_train_59 - 2
    y_valid_59 = y_valid_59 - 2
    subsample_aux = np.random.choice(len(X_train_59), size=(49000,), replace=False)
    subsample = np.random.choice(len(X_train_04), size=(1000,), replace=False)
    return X_train_04[subsample]

def resnet_model_forward(model, x: torch.Tensor) -> torch.Tensor:
    #input x should have dimensions (N_examples, 3, 32, 32)
    to_return = []
    
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    
    to_return.append(x.cpu().detach().numpy())

    x = model.layer1(x)
    to_return.append(x.cpu().detach().numpy())
    x = model.layer2(x)
    to_return.append(x.cpu().detach().numpy())
    x = model.layer3(x)
    to_return.append(x.cpu().detach().numpy())
    x = model.layer4(x)
    to_return.append(x.cpu().detach().numpy())

    x = model.avgpool(x)
    x = torch.flatten(x, 1)
    to_return.append(x.cpu().detach().numpy())
    x = model.fc(x)

    to_return.append(x.cpu().detach().numpy())
    return to_return

def vit_model_forward(model, x, batch_size=128, device='cpu'):
    data_loader = data.DataLoader(data.TensorDataset(x), batch_size=batch_size)
    lst = []
    for _x in tqdm(data_loader):
        to_return = []
        _x = _x[0].to(device)
        out = model._to_words(_x)
        out = model.emb(out)
        if model.is_cls_token:
            out = torch.cat([model.cls_token.repeat(out.size(0),1,1), out],dim=1)
        out = out + model.pos_emb
        to_return.append(out.cpu().detach().numpy())
        for enc in model.enc:
            out = enc(out)
            to_return.append(out.cpu().detach().numpy())
        if model.is_cls_token:
            out = out[:,0]
        else:
            out = out.mean(1)
        out = model.fc(out)
        lst.append(to_return)
    lst = [np.concatenate(l) for l in zip(*lst)]
    return lst

def ensd(x1, x2, batch_size=None):
    if batch_size is not None:
        r1, r2, r3, r4 = 0, 0, 0, 0
        for i in tqdm(range(x1.shape[1]//batch_size)):
            _x1 = x1[:,(batch_size*i):min((batch_size*(i+1)), x1.shape[1])]
            _x2 = x2[:,(batch_size*i):min((batch_size*(i+1)), x1.shape[1])]
            _r1, _r2, _r3, _r4 = ensd(_x1, _x2)
            r1 = r1 + _r1
            r2 = r2 + _r2
            r3 = r3 + _r3
            r4 = r4 + _r4
        n = (x1.shape[1]//batch_size)
        return r1/n, r2/n, r3/n, r4/n
    c1 = x1.T@x1
    c1 = c1/c1.trace()
    gamma1 = c1.trace()**2/(c1@c1).trace()
    c2 = x2.T@x2
    c2 = c2/c2.trace()
    gamma2 = c2.trace()**2/(c2@c2).trace()
    tau = (c1@c2).trace()
    return gamma1*gamma2*tau, gamma1, gamma2, tau

def main(args):
    pathlib.Path(args.save_path).mkdir(parents=True, exist_ok=True)
    if args.model == 'resnet':
        model_pre = torchvision.models.resnet18(pretrained=False, num_classes=100)
        model_post = torchvision.models.resnet18(pretrained=False, num_classes=100)
    if args.model == 'vit':
        model_pre = ViT(num_classes=100)
        model_post = ViT(num_classes=100)
    model_pre.load_state_dict(torch.load(args.load_path_pre, map_location=torch.device(args.device)))
    model_post.load_state_dict(torch.load(args.load_path_post, map_location=torch.device(args.device)))
    for param in model_pre.parameters():
        param.data = param.data * args.scaling
    model_pre = model_pre.to(args.device)
    model_post = model_post.to(args.device)
    print('Getting data')
    x = get_train_data(args.random_seed)
    print('Extracting features')
    with torch.no_grad():
        if args.model == 'resnet':
            feats_pre = resnet_model_forward(model_pre, torch.from_numpy(x).to(args.device))
            feats_post = resnet_model_forward(model_post, torch.from_numpy(x).to(args.device))
        if args.model == 'vit':
            feats_pre = vit_model_forward(model_pre, torch.from_numpy(x), device=args.device)
            feats_post = vit_model_forward(model_post, torch.from_numpy(x), device=args.device)
    print(feats_pre[2].shape)
    print(feats_post[2].shape)
    df = []
    print('Computing ENSD')
    for i, (x_pre, x_post) in tqdm(enumerate(zip(feats_pre, feats_post))):
        df.append(pd.DataFrame({
            'value': ensd(x_pre.reshape(x_pre.shape[0], -1), x_post.reshape(x_post.shape[0], -1), batch_size=args.batch_size),
            'variable': ['ENSD', 'PR Pre', 'PR Post', 'Overlap'],
            'layer': i
        }))
    df = pd.concat(df).reset_index(drop=True)
    df.to_csv(os.path.join(args.save_path, 'df.csv'))

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=None)
    parser.add_argument('--model', choices=['resnet', 'vit'])
    parser.add_argument('--load_path_pre', type=str, required=True)
    parser.add_argument('--load_path_post', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--scaling', type=float, default=1.)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--batch_size', type=int, default=None)
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
