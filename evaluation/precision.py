import torch


def precision(predictions, targets, threshold=0.5, epsilon=1e-7):
    # Convert predictions to binary (0 or 1)
    # pred_binary = (predictions >= threshold).float()
    predicted = torch.argmax(predictions, dim=-1)
    
    # Calculate true positives and false positives
    true_positives = ((predicted == 1) & (targets == 1)).sum().float()
    false_positives = ((predicted == 1) & (targets == 0)).sum().float()
    
    # Calculate precision
    precision = true_positives / (true_positives + false_positives + epsilon)
    
    return precision
