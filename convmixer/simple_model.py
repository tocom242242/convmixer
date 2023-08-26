import torch
import torch.nn as nn
import torch.optim as optim


# シンプルなCNNモデルの定義
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # 3 channels for CIFAR-10
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        # self.fc1 = nn.Linear(8 * 8 * 32, 128)  # Adjusted size for CIFAR-10
        self.fc1 = nn.Linear(1152, 128)  # Adjusted size for CIFAR-10
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = nn.MaxPool2d(2)(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = self.fc2(x)
        return nn.Softmax(dim=1)(x)
