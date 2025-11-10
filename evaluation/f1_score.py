from .recall import recall
from .precision import precision


def f1_score(predictions, targets, threshold=0.5, epsilon=1e-7):

    precision_score = precision(predictions, targets, threshold, epsilon)
    recall_score = recall(predictions, targets, threshold, epsilon)
    
    f1 = 2 * (precision_score * recall_score) / (precision_score + recall_score + epsilon)
    
    return f1