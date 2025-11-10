import torch


def recall(predictions, targets, threshold=0.5, epsilon=1e-7):
    # Convert predictions to binary (0 or 1)
    # pred_binary = (predictions >= threshold).float()
    predicted = torch.argmax(predictions, dim=-1)
    
    # Calculate true positives and false negatives
    true_positives = ((predicted == 1) & (targets == 1)).sum().float()
    false_negatives = ((predicted == 0) & (targets == 1)).sum().float()
    
    # Calculate recall
    recall = true_positives / (true_positives + false_negatives + epsilon)
    
    return recall