from .recall import recall
from .precision import precision


def f1_score(predictions, targets, threshold=0.5, epsilon=1e-7):
    """
    Calculate F1 score for binary classification.
    
    F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Args:
        predictions (torch.Tensor): Model predictions (logits or probabilities)
        targets (torch.Tensor): Ground truth labels (0 or 1)
        threshold (float): Threshold for converting probabilities to binary predictions
        epsilon (float): Small value to avoid division by zero
    
    Returns:
        torch.Tensor: F1 score
    """
    precision_score = precision(predictions, targets, threshold, epsilon)
    recall_score = recall(predictions, targets, threshold, epsilon)
    
    f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score + epsilon)
    
    return f1