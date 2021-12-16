import torch
import torch.nn.functional as F
from train_dataset import KFoldTrainDataset
from torch.utils.data import DataLoader
from machine import Machine
import math

num_epochs = 3


def train_for_validation_batch(device, datasets, validation_batch):
    model = Machine()
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    for epoch in range(num_epochs):
        for train_batch in range(datasets.folds):
            if train_batch is not validation_batch:
                sub_dataset = datasets.get_dataset(train_batch)
                batch_size = 100
                dataloader = DataLoader(sub_dataset, batch_size=batch_size, shuffle=True)
                NUM_BATCHES = int(len(sub_dataset) / batch_size) + 1
                batch_number = 0
                for x, y in dataloader:
                    x = x.to(device)
                    y = y.to(device)
                    optimizer.zero_grad()
                    y_hat = model(x)
                    loss = F.cross_entropy(y_hat, y)
                    loss.backward()
                    optimizer.step()
                    batch_number += 1
                    print(f'Validation Batch: {validation_batch + 1}, Epoch:{epoch + 1} (of {num_epochs}),'
                          f' Training Batch: {train_batch}'
                          f' Dataloader Batch: {batch_number} of ({NUM_BATCHES}), Loss:{loss.item():.4f}')
    return model


def validate(model, device, validation_dataset):
    model.eval()
    dataloader = DataLoader(validation_dataset, batch_size=1000, shuffle=True)
    total_accuracy = 0
    passes = 0
    for x,y in dataloader:
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)
        pred = torch.argmax(y_hat, dim=1, keepdim=True)
        correct = pred.eq(y.data.view_as(pred)).sum()
        total = x.shape[0]
        accuracy = correct/total
        total_accuracy += accuracy
        passes += 1
    return total_accuracy / passes


def train(device):
    total_accuracy = 0
    min_accuracy = math.inf
    max_accuracy = -math.inf
    best_model = None

    datasets = KFoldTrainDataset()

    for validation_batch in range(datasets.folds):
        model = train_for_validation_batch(device, datasets, validation_batch)
        validation_dataset = datasets.datasets[validation_batch]
        accuracy = validate(model, device, validation_dataset)

        total_accuracy += accuracy

        if accuracy > max_accuracy:
            max_accuracy = accuracy
            best_model = model

        if accuracy < min_accuracy:
            min_accuracy = accuracy

    print(f"Min Accuracy: {min_accuracy}")
    print(f"Max Accuracy: {max_accuracy}")
    print(f"Average Accuracy: {total_accuracy/datasets.folds}")

    torch.save(best_model, 'models/cnn_trans.h5')


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(device)