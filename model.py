import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolution layer 1: input = 1 channel, output = 8 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1)  # output: 8x64x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: 8x32x32

        # Convolution layer 2: input = 8, output = 16
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)  # output: 16x32x32
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)       # output: 16x16x16

        # Fully connected layer: 16 * 16 * 16 → 64
        self.fc1 = nn.Linear(16 * 16 * 16, 64)
        self.fc2 = nn.Linear(64, 5)  # 5 shape classes

    def forward(self, x):
        x = F.relu(self.conv1(x))  # [batch, 8, 64, 64]
        x = self.pool1(x)          # [batch, 8, 32, 32]
        x = F.relu(self.conv2(x))  # [batch, 16, 32, 32]
        x = self.pool2(x)          # [batch, 16, 16, 16]
        
        x = x.view(x.size(0), -1)  # flatten for FC layer

        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x