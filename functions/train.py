from copy import deepcopy

import torch
import torch.optim as optim
import torch.nn.functional as F


def train(model, train_data, val_data, test_every_n_epochs=50, epochs=1000, lr=0.01, momentum=0., lr_tuning=True, test_at_end_only=False, threshold=1e-5):
    or_model = deepcopy(model)
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    losses = []
    test_preds = []
    x, y = train_data
    val_x, val_y = val_data
    loss = float('Inf')
    i = 0
    while (loss > threshold) and (i<=epochs):
        i += 1
        optimizer.zero_grad()
        loss = F.mse_loss(model(x), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        loss = loss.item()
        if (i%test_every_n_epochs==0):
            with torch.no_grad():
                new_df = pd.DataFrame({
                    'loss': F.mse_loss(model(val_x), val_y).numpy()
                })
                new_df['epoch'] = i
                test_preds.append(new_df)
        if lr_tuning and ((loss > 1) | np.isnan(loss)):
            lr = lr/10
            print(f'Decreasing learning rate to {lr}')
            return train(or_model, train_data, val_data, test_every_n_epochs=test_every_n_epochs, epochs=epochs, lr=lr, momentum=momentum, lr_tuning=lr_tuning, test_at_end_only=test_at_end_only, threshold=threshold)
    with torch.no_grad():
        new_df = pd.DataFrame({
            'loss': F.mse_loss(model(val_x), val_y).numpy()
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
    ]).reset_index(drop=True)