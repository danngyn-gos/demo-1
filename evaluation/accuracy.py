import torch


def accuracy(predictions, targets):
    predicted = torch.argmax(predictions, dim=-1)
    correct = (predicted == targets).sum().item()
    return correct / targets.size(0)