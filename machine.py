import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Machine(nn.Module):
    def __init__(self):
        super().__init__()
        self.TRADITIONAL_TRANSFER_LEARNING = False
        self.resnet = torchvision.models.resnet18(pretrained=True)

        if self.TRADITIONAL_TRANSFER_LEARNING:
            number_input = self.resnet.fc.in_features
            self.resnet.fc = nn.Linear(number_input, 2)
        else:
            number_input = self.resnet.fc.out_features
            self.fc = nn.Linear(number_input, 2)


    def forward(self, x):
        x = self.resnet(x)
        if not self.TRADITIONAL_TRANSFER_LEARNING:
            x = F.relu(x)
            x = self.fc(x)
        return F.log_softmax(x, dim=1)
