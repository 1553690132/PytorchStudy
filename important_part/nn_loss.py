import torch
from torch.nn import L1Loss, MSELoss, CrossEntropyLoss

input_v = torch.tensor([1, 2, 3], dtype=torch.float)
target = torch.tensor([4, 5, 6], dtype=torch.float)

input_v = torch.reshape(input_v, (1, 1, 1, 3))
target = torch.reshape(target, (1, 1, 1, 3))

loss = L1Loss()
result = loss(input_v, target)
print(result)

loss2 = MSELoss()
result2 = loss2(input_v, target)

print(result2)

# x = torch.tensor([0.1, 0.2, 0.3])
# y = torch.tensor([1])
# x = torch.reshape(x, (1, 3))
#
loss_cross = CrossEntropyLoss()
# result3 = loss_cross(x, y)
# print(result3)


x = torch.tensor([0.1, 0.1, 0.1])
y = torch.tensor([1])
x = torch.reshape(x, (1, 3))
print(loss_cross(x, y))
