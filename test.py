import torch
from torch.utils.data import DataLoader
import test_dataset

def test(device):
    cid = test_dataset.TestDataset()
    dataloader = DataLoader(cid.dataset, batch_size=1000, shuffle=False)
    model = torch.load("models/cnn_trans.h5")
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    for (x, y) in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        pred = torch.argmax(y_hat, dim=1, keepdim=True)
        correct += pred.eq(y.data.view_as(pred)).sum()
        total += x.shape[0]
        print(f'Total:{total}, Correct:{correct}, Accuracy:{correct/total*100:.2f}')

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test(device)
