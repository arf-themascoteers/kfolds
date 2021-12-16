import torch
from dataset import CustomImageDataset
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


def test(device):
    batch_size = 1
    cid = CustomImageDataset(is_train=False)
    dataloader = DataLoader(cid, batch_size=batch_size, shuffle=False)
    model = torch.load("models/cnn_trans.h5")
    model.to(device)
    x = None
    y = None
    count = 0
    for (x, y) in dataloader:
        z = x.cpu()
        z = z.squeeze()
        z = torch.permute(z, (1, 2, 0))
        z = z.detach().numpy()
        plt.imshow(z)
        plt.show()

        x = x.to(device)
        x.requires_grad = True
        y = y.to(device)

        if y[0] != 1:
            continue

        y_hat = model(x)
        index = y_hat.argmax()
        final = y_hat[0,index]
        final.backward()
        x = x.grad
        x = torch.abs(x)
        max_val = torch.max(x)
        scale = float(255/max_val)
        #x = x * scale
        x = x.squeeze()
        x = torch.permute(x, (1, 2, 0))
        x = torch.max(x, dim=2)[0]
        x = x.cpu().detach().numpy()
        plt.imshow(x, cmap="hot")
        plt.show()
        count += 1
        if count == 10:
            break
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
