# xの二乗の関係を学習させようとしてみる例
# データ数を増やすことで、ある程度正解に近い値が出るようになった
import torch

# y = x1 ** 2 + x2 ** 2 + 5
x = torch.tensor([[1.0, 2.0], [2.0, 3.0], [3.0, 3.0], [4.0, 1.0], [5.0, -3.0], [2.0, 0.0], [0.0, 4.0], [2.0, 1.0], [3.0, 2.0]])
y = torch.tensor([[10.0], [18.0], [23.0], [22.0], [39.0], [9.0], [21.0], [10.0], [18.0]])

model = torch.nn.Sequential(
  torch.nn.Linear(in_features = 2, out_features = 16),
  torch.nn.ReLU(), #活性化関数
  torch.nn.Linear(in_features = 16, out_features = 1),
)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

model.train()

for epoch in range(1000):
	pred_y = model(x)
	loss = loss_fn(pred_y, y)

	optimizer.zero_grad()
	loss.backward()
	optimizer.step()

test_x = torch.tensor([[1.0, 4.0], [2.0, 4.0]])
test_y = model(test_x)
true_y = torch.tensor([[1.0 ** 2 + 4.0 ** 2 + 5], [2.0 ** 2 + 4.0 ** 2 + 5]])
print ("正解は", true_y)
print ("予測は", test_y)
