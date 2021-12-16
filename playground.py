import torch
import torch.nn as nn
from torchvision import datasets, transforms
import cnn

data = torch.randn(1, 64,1,1,dtype=torch.cfloat)
print(data.shape)
model = nn.Sequential(
     nn.Flatten()
)

output = model(data)
print(output.shape)