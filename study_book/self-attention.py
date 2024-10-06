import torch

value_len = 3
num_attention_heads = 12
hidden_size = 768

Query = torch.randn(value_len, hidden_size)
Value = torch.randn(value_len, hidden_size)
Key = torch.randn(value_len, hidden_size)

Query = torch.reshape(Query, (value_len, num_attention_heads, hidden_size // num_attention_heads))
Query = torch.transpose(Query, 0, 1)

Value = torch.reshape(Value, (value_len, num_attention_heads, hidden_size // num_attention_heads))
Value = torch.transpose(Value, 0, 1)

Key = torch.reshape(Key, (value_len, num_attention_heads, hidden_size // num_attention_heads))
Key = torch.transpose(Key, 0, 1)

scores = Query @ torch.permute(Key, (0, 2, 1))
scores = torch.softmax(scores, dim=-1)
print(scores.shape)
out = scores @ Value
print(out.shape)
out = torch.permute(out, (1, 0, 2))
print(out.shape)
out = torch.reshape(out, (value_len, hidden_size))
print(out.shape)
