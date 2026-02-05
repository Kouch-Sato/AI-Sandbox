# xの次元数を増やしているが、中間層は用意していない。
import torch

# y = 3 * x1 - 2 * x2 + 2
x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 10.0], [4.0, 1.0], [5.0, -3.0]])
y = torch.tensor([[1.0], [2.0], [-9.0], [12.0], [23.0]])

model = torch.nn.Linear(in_features = 2, out_features = 1)
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

model.train()

for epoch in range(1000):
	pred_y = model(x)
	loss = loss_fn(pred_y, y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

print (model.weight, model.bias.item())
