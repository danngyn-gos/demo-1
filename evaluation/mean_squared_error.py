import torch


def mean_squared_error(preds: torch.Tensor,
                       targets: torch.Tensor):
    n = preds.shape[0]
    
    mse = torch.sum((targets-preds)**2)/n
    
    return mse


def root_mean_squared_error(preds: torch.Tensor,
                            targets: torch.Tensor):
    n = preds.shape[0]
    
    mse = torch.sum((targets-preds)**2)/n
    
    return torch.sqrt(mse)