import torch
import torch.nn as nn
import torch.nn.functional as F


class BonusNetwork(nn.Module):
    def __init__(self):
        super(BonusNetwork, self).__init__()
        self.conv0 = torch.nn.Conv2d(
                in_channels=1,
                out_channels=3,
                kernel_size=3,
                padding=2,
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=6,
                kernel_size=5,
                stride=3,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=5, stride=3),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=6,
                out_channels=9,
                kernel_size=5,
                stride=2,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.drop = F.dropout
        self.fc1 = torch.nn.Linear(81, 24)
        self.fc2 = torch.nn.Linear(24, 3)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = self.fc2(x)
        return x


class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(
                in_channels=1,
                out_channels=6,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=16),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=6,
                out_channels=16,
                kernel_size=3,
                stride=2,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=8),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.fc1 = torch.nn.Linear(960, 256)
        self.fc2 = torch.nn.Linear(256, 64)
        self.fc3 = torch.nn.Linear(64, 10)
        self.drop = F.dropout

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.drop(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class EpicNetwork(nn.Module):
    def __init__(self):
        super(EpicNetwork, self).__init__()
        self.conv1 = nn.Conv1d(
                in_channels=1,
                out_channels=128,
                kernel_size=3,
                stride=3,
                padding=4,
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                stride=3,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=3,
                stride=2,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv4 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv6 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=256,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(
                in_channels=256,
                out_channels=512,
                kernel_size=3,
                stride=1,
                padding=0,
            ),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.fc0 = nn.Linear(1024, 256)
        self.fc1 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = F.dropout(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc0(x))
        x = self.fc1(x)
        return x
