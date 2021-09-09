import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


def grid_2d(w, h):
    uv = np.mgrid[:w, h-1:-1:-1].T
    uv = torch.from_numpy(uv)
    uv1 = torch.cat([uv, torch.ones_like(uv)[..., :1]], dim=-1)
    uv1 = torch.matmul(uv1, torch.inverse(K).T)

    return uv1