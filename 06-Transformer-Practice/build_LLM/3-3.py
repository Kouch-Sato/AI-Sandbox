import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

token_count = inputs.shape[0] 
attention_scores = torch.empty(token_count, token_count) # torch.Size([6, 6])
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attention_scores[i][j] = torch.dot(x_i, x_j)

attention_weights = torch.softmax(attention_scores, dim=-1) 

context_vectors = torch.empty(inputs.shape) # torch.Size([6, 3])
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        context_vectors[i] += attention_weights[i][j] * x_j

print(context_vectors)
