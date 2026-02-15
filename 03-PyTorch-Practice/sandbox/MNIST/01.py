
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_datasets = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_datasets  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_datasets, batch_size=256, shuffle=False, num_workers=0)

x, y = train_datasets[10]
print(x)  # torch.Size([1, 28, 28]) 5
x.show()

# model = torch.nn.Sequential(
#   torch.nn.Linear(in_features = 2, out_features = 16),
#   torch.nn.ReLU(), #活性化関数
#   torch.nn.Linear(in_features = 16, out_features = 1),
# )

# loss_fn = torch.nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

# model.train()

# for epoch in range(1000):
# 	pred_y = model(x)
# 	loss = loss_fn(pred_y, y)

# 	optimizer.zero_grad()
# 	loss.backward()
# 	optimizer.step()

# test_x = torch.tensor([[1.0, 4.0], [2.0, 4.0]])
# test_y = model(test_x)
# true_y = torch.tensor([[1.0 ** 2 + 4.0 ** 2 + 5], [2.0 ** 2 + 4.0 ** 2 + 5]])
# print ("正解は", true_y)
# print ("予測は", test_y)
