import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

x_2 = inputs[1]
d_in = inputs.shape[1]
d_out = 2 

torch.manual_seed(123)
w_query = torch.nn.Parameter(torch.rand(d_in, d_out))
w_key = torch.nn.Parameter(torch.rand(d_in, d_out))
w_value = torch.nn.Parameter(torch.rand(d_in, d_out))

query_2 = x_2 @ w_query

keys = inputs @ w_key
values = inputs @ w_value

attention_scores_2 = query_2 @ keys.T

d_k = keys.shape[1]
attention_weights_2 = torch.softmax(attention_scores_2 / (d_k ** 0.5), dim=-1)

context_vector_2 = attention_weights_2 @ values

print(keys.shape)
print(values.shape)

print(attention_weights_2)
print(context_vector_2)
