import torch

# y = - 3x + 10
x = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
y = torch.tensor([[7.0], [4.0], [1.0], [-2.0], [-5.0]])

model = torch.nn.Linear(in_features = 1, out_features = 1)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

model.train()

for epoch in range(1000):
	pred_y = model(x)
	loss = loss_fn(pred_y, y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


print (model.weight.item(), model.bias.item())
