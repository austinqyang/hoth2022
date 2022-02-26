import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, kernel_size = 5, padding = 2)
        self.conv2 = nn.Conv2d(4, 16, kernel_size = 3, padding = 1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size = 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(2304, 512)
        self.fc2 = nn.Linear(512, 64)
        self.fc3 = nn.Linear(64, 7)

    def forward(self, x):
        #(n, 1, 48, 48)
        x = self.conv1(x)
        x = F.relu(x)
        #(n, 4, 48, 48)
        x = self.pool(x)
        #(n, 4, 24, 24)
        x = self.conv2(x)
        x = F.relu(x)
        #(n, 16, 24, 24)
        x = self.pool(x)
        #(n, 16, 12, 12)
        x = self.conv3(x)
        x = F.relu(x)
        #(n, 64, 12, 12)
        x = self.pool(x)
        #(n, 64, 6, 6)
        x = torch.reshape(x, (-1, 64 * 6 * 6))
        #(n, 64 * 6 * 6)
        x = F.relu(self.fc1(x))
        #(n, 512)
        x = F.relu(self.fc2(x))
        #(n, 64)
        x = self.fc3(x)
        #(n, 7)
        return x