import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.w_query = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_key = torch.nn.Parameter(torch.rand(d_in, d_out))
        self.w_value = torch.nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.w_key
        queries = x @ self.w_query
        values = x @ self.w_value

        attention_scores = queries @ keys.T
        d_k = keys.shape[1]
        attention_weights = torch.softmax(attention_scores / (d_k ** 0.5), dim=-1)

        context_vectors = attention_weights @ values
        return context_vectors


inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

d_in = inputs.shape[1]
d_out = 2 

torch.manual_seed(123)

self_attention = SelfAttention_v1(d_in, d_out)
context_vectors = self_attention(inputs)

print(self_attention)
print(context_vectors.shape)
print(context_vectors)
