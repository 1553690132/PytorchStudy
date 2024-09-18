import torch
from torch import nn

model = nn.Linear(1, 1)

x = torch.randn(100, 1)

y = 3 * x + 1 + torch.randn(100, 1) * 0.1

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

for epoch in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f'Epoch {epoch + 1}, Loss: {loss.item():.4f}')
