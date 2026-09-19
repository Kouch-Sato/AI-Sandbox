# MNISTの前半
# https://zenn.dev/kouch/articles/a28316bffbdba8

# MNISTの後半


import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.ToTensor()
train_datasets = datasets.MNIST(root="data", train=True, download=True, transform=transform)
test_datasets  = datasets.MNIST(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_datasets, batch_size=64, shuffle=True, num_workers=0)
test_loader  = DataLoader(test_datasets, batch_size=256, shuffle=False, num_workers=0)

model = torch.nn.Sequential(
    torch.nn.Flatten(),
    torch.nn.Linear(in_features = 28 * 28, out_features = 128),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features = 128, out_features = 64),
    torch.nn.ReLU(),
    torch.nn.Linear(in_features = 64, out_features = 10),
)

# 回帰ではなく分類なので、交差エントロピーを使う
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01)

model.train()

for epoch in range(300):
    for x, y in train_loader:
        pred_y = model(x)
        loss = loss_fn(pred_y, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()

correct = 0
total = 0

for x, y in test_loader:
    pred_y = model(x).argmax(dim = 1)
    correct += (pred_y == y).sum().item()
    total += len(y)

accuracy = correct / total
print (f"{accuracy * 100}%")
