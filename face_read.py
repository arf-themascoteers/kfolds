import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset

img_dir = "data/faces"
labels = pd.read_csv("data/test.csv")
print(labels.iloc[0,1])
