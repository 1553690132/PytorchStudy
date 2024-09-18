import time

import torch
from torch import nn
from torch.utils.data import DataLoader


def train_data(optimizer_fn, optimizer_hyperparameters, features, labels, batch_size=10, num_epoch=2):
    net = nn.Sequential(
        nn.Linear(features.shape[-1], 1),
    )
    loss = nn.MSELoss()
    optimizer = optimizer_fn(net.parameters(), **optimizer_hyperparameters)

    def eval_loss():
        return loss(net(features).view(-1), labels).item() / 2

    ls = [eval_loss()]

    data_iter = DataLoader(torch.utils.data.TensorDataset(features, labels), batch_size=batch_size, shuffle=True)

    for _ in range(num_epoch):
        start = time.time()
        for batch_i, (x, y) in enumerate(data_iter):
            l = loss(net(x).view(-1), y) / 2

            optimizer.zero_grad()
            l.backward()
            optimizer.step()

            if (batch_i + 1) * batch_size % 100 == 0:
                ls.append(eval_loss())
        print(f"loss: {ls[-1]} {time.time() - start} sec per epoch")
