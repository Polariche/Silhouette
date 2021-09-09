import torch
import torch.nn.functional as F
import torch.nn as nn

def sync_end_dim(x, y):
    if len(x.shape) < len(y.shape):
        k = len(y.shape) - len(x.shape)

        for i in range(k):
            x = x.unsqueeze(0)

    return x