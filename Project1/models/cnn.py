import torch.nn as nn
import torch


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.Conv1 = nn.Conv2d(3, 32, 3, padding = 1) #(32,32,32)
        self.Pool = nn.MaxPool2d(2, 2) #(32,16,16)
        self.Conv2 = nn.Conv2d(32, 64, 3, padding = 1) #(64,16,16)
        self.Conv3 = nn.Conv2d(64, 64, 3, padding = 1) #
        self.fc1 = nn.Linear(64*8*8,64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.Pool(torch.relu(self.Conv1(x)))
        x = self.Pool(torch.relu(self.Conv2(x)))
        x = torch.relu(self.Conv3(x))
        x = x.view(-1, 64*8*8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x