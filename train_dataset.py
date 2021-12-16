import os

import PIL.Image
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import random
import dataset


class KFoldTrainDataset:
    def __init__(self, folds = 10):
        self.folds = folds
        self.file = "data/train.csv"
        self.img_labels = pd.read_csv(self.file)

        total_images = len(self.img_labels.index)
        train_batch = int(total_images / self.folds)
        train_images = train_batch * self.folds
        self.indices = list(range(train_images))
        random.shuffle(self.indices)
        self.datasets = []
        for i in range(self.folds):
            x_i = []
            y_i = []
            start = i * train_batch
            end = start + train_batch
            for index in range(start, end):
                row_number = self.indices[index]
                row = self.img_labels.iloc[row_number]
                x_i.append(row.iloc[0])
                y_i.append(row.iloc[1])

            self.datasets.append(dataset.CustomImageDataset(x_i, y_i))

    def get_dataset(self, batch_no):
        return self.datasets[batch_no]

if __name__ == "__main__":
    kfd = KFoldTrainDataset()
    dataloader = DataLoader(kfd.datasets[0], batch_size=50, shuffle=True)

    for image, label in dataloader:
        print(image.shape)
        print(label)
        for i in image:
            plt.imshow(i[0].numpy())
            plt.show()
        print(image[0])
        exit(0)
