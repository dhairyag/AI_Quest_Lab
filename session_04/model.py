import torch
import torch.nn as nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self, kernel_config=[32, 64, 64, 64]):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, kernel_config[0], 3, 1)
        self.conv2 = nn.Conv2d(kernel_config[0], kernel_config[1], 3, 1)
        self.conv3 = nn.Conv2d(kernel_config[1], kernel_config[2], 3, 1)
        self.conv4 = nn.Conv2d(kernel_config[2], kernel_config[3], 3, 1)
        fc1_input = kernel_config[3] * 4 * 4  # Calculate based on input size and operations
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(fc1_input, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1) 