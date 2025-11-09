def recall(predictions, targets, threshold=0.5, epsilon=1e-7):
    # Convert predictions to binary (0 or 1)
    pred_binary = (predictions >= threshold).float()
    
    # Calculate true positives and false negatives
    true_positives = ((pred_binary == 1) & (targets == 1)).sum().float()
    false_negatives = ((pred_binary == 0) & (targets == 1)).sum().float()
    
    # Calculate recall
    recall = true_positives / (true_positives + false_negatives + epsilon)
    
    return recall