import torch
import torch.nn as nn

class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, bias)
            for i in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, bias=False):
        super().__init__()
        self.w_query = nn.Linear(d_in, d_out, bias=bias)
        self.w_key = nn.Linear(d_in, d_out, bias=bias)
        self.w_value = nn.Linear(d_in, d_out, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        batch_size, num_tokens, d_in = x.shape

        keys = self.w_key(x)
        queries = self.w_query(x)
        values = self.w_value(x)

        attention_scores = queries @ keys.transpose(1, 2) # batchが追加されたから、0次元は変えずに、1次元と2次元だけ変えたい

        attention_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        
        d_k = keys.shape[-1]
        self.attention_weights = torch.softmax(attention_scores / (d_k ** 0.5), dim=-1)

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

torch.manual_seed(123)

batch = torch.stack((inputs, inputs), dim=0)
context_length = batch.shape[1]
multi_head_attention = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=3)
context_vec = multi_head_attention.forward(batch)

print(context_vec.shape)
print(context_vec)
