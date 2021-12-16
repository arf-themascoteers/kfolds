import os

import PIL.Image
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import dataset


class TestDataset:
    def __init__(self):
        self.file = "data/test.csv"
        self.img_labels = pd.read_csv(self.file)

        total_images = len(self.img_labels.index)
        x = []
        y = []
        for row_number in range(total_images):
            row = self.img_labels.iloc[row_number]
            x.append(row.iloc[0])
            y.append(row.iloc[1])

        self.dataset = dataset.CustomImageDataset(x, y)

    def get_dataset(self):
        return self.dataset

if __name__ == "__main__":
    kfd = TestDataset()
    dataloader = DataLoader(kfd.dataset, batch_size=50, shuffle=True)

    for image, label in dataloader:
        print(image.shape)
        print(label)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
        print(image[0])
        exit(0)
