import torch


def accuracy(preds, labels):
    predicted = torch.argmax(preds, dim=-1)
    correct = (predicted == labels).sum().item()
    return correct / labels.size(0)