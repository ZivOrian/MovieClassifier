import torch
import numpy as np


def get_indices(lst: list, targets):
    """Get the indices of targets in lst"""
    indices = []
    for target in targets:
        if target in lst:
            indices.append(lst.index(target))
    return indices


def topK_one_hot(loc_list: list, num_classes: int):
    """
    Create one-hot encoding from list of class indices.
    Returns a list (not tensor) for compatibility.
    """
    # Handle empty list case
    if len(loc_list) == 0:
        return [0.0 for i in range(num_classes)]
    
    one_hot_tensor = [1.0 if i in loc_list else 0.0 for i in range(num_classes)]
    return one_hot_tensor


def smooth_multi_hot(multi_hot, num_valid_labels=None, eps=1e-5):
    """
    Apply label smoothing to multi-hot encoded labels.
    
    Args:
        multi_hot: torch.Tensor or list of binary labels
        num_valid_labels: Number of positive labels (for reference, not used in calculation)
        eps: Smoothing parameter
    
    Returns:
        torch.Tensor with smoothed probabilities
    """
    if not isinstance(multi_hot, torch.Tensor):
        multi_hot = torch.tensor(multi_hot, dtype=torch.float32)
    
    # ✅ CRITICAL FIX: Check if all zeros (no valid labels)
    if multi_hot.sum() == 0:
        # Return uniform distribution when no labels exist
        num_classes = len(multi_hot)
        return torch.ones(num_classes, dtype=torch.float32) / num_classes
    
    # Add eps everywhere to avoid zeros
    smoothed = multi_hot + eps
    
    # Normalize so sum = 1
    sum_val = smoothed.sum()
    
    # ✅ Safety check: prevent division by zero
    if sum_val == 0 or torch.isnan(sum_val) or torch.isinf(sum_val):
        num_classes = len(multi_hot)
        return torch.ones(num_classes, dtype=torch.float32) / num_classes
    
    smoothed = smoothed / sum_val
    
    # ✅ Final validation: ensure no NaN/Inf and values in valid range
    if torch.isnan(smoothed).any() or torch.isinf(smoothed).any():
        num_classes = len(multi_hot)
        return torch.ones(num_classes, dtype=torch.float32) / num_classes
    
    # Clamp to prevent exact 0 or 1 (which can cause log(0) in BCE)
    epsilon = 1e-7
    smoothed = torch.clamp(smoothed, min=epsilon, max=1.0-epsilon)
    
    return smoothed
