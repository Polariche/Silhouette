import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def grid_2d(w, h, K=None):
    if K is None:
        half = 0.5*(w+h)
        hhalf = 0.5*half
        K = torch.tensor([[half, 0, hhalf],
                            [0, half, hhalf],
                            [0, 0, 1]])

    uv = np.mgrid[:w, h-1:-1:-1].T
    uv = torch.from_numpy(uv).float()
    uv1 = torch.cat([uv, torch.ones_like(uv)[..., :1]], dim=-1)
    uv1 = torch.matmul(uv1, torch.inverse(K).T)

    uv = uv1[..., :2].reshape(-1, 2)

    return uv