import torch
import torch.nn.functional as F
import torch.nn as nn
import utils

class SphereSDF(nn.Module):
    def __init__(self, pos, r):
        super(SphereSDF, self).__init__()
    
    def forward(self, x):
        assert x.shape[-1] == self.pos.shape[-1]
        
        pos = utils.sync_end_dim(self.pos)
        x = x - pos
        s = torch.sqrt((x**2).sum(dim=-1, keepdim=True)) - self.r

        return s
        