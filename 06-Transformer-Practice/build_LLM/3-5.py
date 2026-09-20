import torch
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out, bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=bias)
        self.w_key = nn.Linear(d_in, d_out, bias=bias)
        self.w_value = nn.Linear(d_in, d_out, bias=bias)

    def forward(self, x):
        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        attention_scores = queries @ keys.T
        context_length = attention_scores.shape[0]

        mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
        masked_attention_scores = attention_scores.masked_fill(mask.bool(), -torch.inf)
        
        d_k = keys.shape[1]
        self.attention_weights = torch.softmax(masked_attention_scores / (d_k ** 0.5), dim=-1)

        context_vectors = self.attention_weights @ values
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

torch.manual_seed(789)

self_attention = SelfAttention_v1(d_in, d_out)
context_vectors = self_attention(inputs)

print(self_attention.attention_weights)
print(context_vectors.shape)
