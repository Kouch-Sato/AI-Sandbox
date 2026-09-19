# Transformerの概念理解
import torch
import torch.nn.functional as F

X = torch.tensor([
    [1.0, 0.0],
    [0.2, 0.8],
    [0.1, 0.9]
])

Q = X
K = X
V = X

# .Tは転置, @は行列の積 つまりQとKの内積を計算している
scores = Q @ K.T
attention = F.softmax(scores, dim = -1)
output = attention @ V

print("scores:", scores)
print("attention:", attention) 
print("output:", output) 
