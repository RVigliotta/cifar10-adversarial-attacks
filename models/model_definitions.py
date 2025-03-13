import torch.nn as nn


class TestCNN(nn.Module):
    def __init__(self):
        super(TestCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(32 * 15 * 15, 10)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = x.view(-1, 32 * 15 * 15)
        x = self.fc1(x)
        return x
