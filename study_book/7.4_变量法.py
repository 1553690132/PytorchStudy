import torch
from torch import nn

model = nn.Linear(1, 1)

# momentum代表动量的超参数
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

loss_fn = nn.MSELoss()

x = torch.randn(100, 1)
# y = 3x+1+噪声 模拟真实环境的不稳定数据
y = 3 * x + 1 + torch.randn(100, 1) * 0.1

for epoch in range(100):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"{epoch} step, loss{loss:.4f}")
