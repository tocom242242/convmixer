import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from convmixer.convmixer import ConvMixer
from convmixer.simple_model import SimpleCNN

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_size = 1000
test_size = 1000

transform_cifar = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
        ),  # 3 channels for CIFAR-10
    ]
)

train_dataset_cifar = datasets.CIFAR10(
    root="./data", train=True, download=True, transform=transform_cifar
)
test_dataset_cifar = datasets.CIFAR10(
    root="./data", train=False, transform=transform_cifar
)

train_indices_cifar = torch.randperm(len(train_dataset_cifar))[:train_size]
test_indices_cifar = torch.randperm(len(test_dataset_cifar))[:test_size]

train_subset_cifar = Subset(train_dataset_cifar, train_indices_cifar)
test_subset_cifar = Subset(test_dataset_cifar, test_indices_cifar)

train_loader_cifar = DataLoader(train_subset_cifar, batch_size=32, shuffle=True)
test_loader_cifar = DataLoader(test_subset_cifar, batch_size=32, shuffle=False)


def main():
    model_cifar = ConvMixer(3, 3)
    # 損失関数と最適化手法を定義
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_cifar.parameters(), lr=0.001)
    # モデルのインスタンスを作成

    # トレーニングと評価を実行
    train(model_cifar, train_loader_cifar, criterion, optimizer, epochs=5)
    accuracy_cifar = evaluate(model_cifar, test_loader_cifar)
    accuracy_cifar


# トレーニングの関数を定義
def train(model, train_loader, criterion, optimizer, epochs=1):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {total_loss/len(train_loader)}")


# モデルの評価関数を定義
def evaluate(model, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            outputs = model(data)
            _, predicted = outputs.max(1)
            correct += predicted.eq(target).sum().item()
    accuracy = 100.0 * correct / len(test_loader.dataset)
    print(f"Accuracy: {accuracy}%")
    return accuracy


main()
