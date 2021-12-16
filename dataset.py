import os

import PIL.Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms



class CustomImageDataset(Dataset):
    def __init__(self, x, y):
        self.img_dir = "data/faces"
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(178),
            transforms.Resize(64),
        ])
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.x[idx])
        image = PIL.Image.open(img_path)
        image = self.transforms(image)
        label = self.y[idx]
        return image, torch.tensor(label)
