import torch
from torch import nn
import torch.nn.functional as F

num_hiddens = 256
vocab_size = 10000

rnn_layer = nn.RNN(input_size=vocab_size, hidden_size=num_hiddens)

num_steps = 35
batch_size = 2
state = None
x = torch.rand(num_steps, batch_size, vocab_size)
y, state_new = rnn_layer(x, state)
print(y.shape, len(state_new), state_new[0].shape)


class RNNModel(nn.Module):
    def __init__(self, rnn_layer, vocab_size):
        super(RNNModel, self).__init__()
        self.rnn = rnn_layer
        self.hidden_size = rnn_layer.hidden_size * (2 if rnn_layer.bidirectional else 1)
        self.vocab_size = vocab_size
        self.dense = nn.Linear(self.hidden_size, self.vocab_size)
        self.state = None

    def forward(self, input, state):
        # 获取one-hot向量表示
        x = F.one_hot(input, self.vocab_size).float()
        y, self.state = self.rnn(x, state)
        output = self.dense(y.view(-1, y.shape[-1]))
        return output, self.state
