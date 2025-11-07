import torch


def mean_squared_error(preds: torch.Tensor,
                       targets: torch.Tensor):

    return ((targets-preds)**2).mean()


def root_mean_squared_error(preds: torch.Tensor,
                            targets: torch.Tensor):
    
    return torch.sqrt(((targets-preds)**2).mean())