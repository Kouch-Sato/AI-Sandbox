import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, bias=False):
        super().__init__()

        assert (d_out % num_heads == 0), "割り切れないぜ"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.w_query = nn.Linear(d_in, d_out, bias=bias)
        self.w_key = nn.Linear(d_in, d_out, bias=bias)
        self.w_value = nn.Linear(d_in, d_out, bias=bias)

        self.out_proj = nn.Linear(d_out, d_out)
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

        keys = keys.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        attention_scores = queries @ keys.transpose(2, 3)
        attention_scores.masked_fill_(self.mask[:num_tokens, :num_tokens].bool(), -torch.inf)
        
        d_k = keys.shape[-1]
        attention_weights = torch.softmax(attention_scores / (d_k ** 0.5), dim=-1)
        attention_weights = self.dropout(attention_weights)

        context_vectors = (attention_weights @ values).transpose(1, 2)
        context_vectors = context_vectors.contiguous().view(
            batch_size, num_tokens, self.d_out
        )
        context_vectors = self.out_proj(context_vectors)

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
multi_head_attention = MultiHeadAttention(d_in, d_out, context_length, dropout=0.0, num_heads=2)
context_vec = multi_head_attention.forward(batch)

print(context_vec.shape)
print(context_vec)
