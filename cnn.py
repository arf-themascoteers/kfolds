import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.machine = nn.Sequential(
            nn.Conv2d(1, 16, (3,3), padding=1), # -> N, 16, 14, 14
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(16, 64, (3,3),padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(3136, 1000),
            nn.ReLU(),
            nn.Linear(1000, 10)
        )


    def forward(self, x):
        x = self.machine(x)
        return F.log_softmax(x, dim=1)
