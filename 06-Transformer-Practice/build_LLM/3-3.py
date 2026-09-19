import torch

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], # Your     (x^1)
   [0.55, 0.87, 0.66], # journey  (x^2)
   [0.57, 0.85, 0.64], # starts   (x^3)
   [0.22, 0.58, 0.33], # with     (x^4)
   [0.77, 0.25, 0.10], # one      (x^5)
   [0.05, 0.80, 0.55]] # step     (x^6)
)

query = inputs[1]
attention_scores_2 = torch.empty(inputs.shape[0])
for index, value in enumerate(inputs):
    attention_scores_2[index] = torch.dot(value, query)

attention_weights_2 = torch.softmax(attention_scores_2, dim=0) # torch.Size([6])

print(attention_weights_2.shape)

context_vector_2 = torch.empty(query.shape) # torch.Size([3])
for index, value in enumerate(inputs):
    context_vector_2 += attention_weights_2[index] * value

print(query)
print(context_vector_2)