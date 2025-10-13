import torch
import numpy as np


def get_indices(lst: list, targets): # Made for getting the classes of the movie genres
    indices = []
    for target in targets:
        if target in lst:
            indices.append(lst.index(target))
    return indices


def topK_one_hot(loc_list: list, num_classes: int) -> torch.Tensor:
    one_hot_tensor = [0]*num_classes
    one_hot_tensor = [1. if i in loc_list else 0. for i in range(19)]
    return one_hot_tensor


def smooth_multi_hot(multi_hot, eps=1e-3):
    if not isinstance(multi_hot, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")

    # add eps everywhere to avoid zeros
    smoothed = multi_hot + eps  

    # normalize so sum = 1
    smoothed = smoothed / smoothed.sum()

    return smoothed


