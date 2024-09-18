import torch
from torch import nn

model = nn.Sequential(nn.Linear(10, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid())

loss_fn = nn.MSELoss()

optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-8)

inputs = torch.randn(100, 10)
labels = torch.randn(100, 1)

for epoch in range(100):
    grad = model(inputs)
    loss = loss_fn(grad, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item():.4f}')


