import torch
import torch.nn.functional as F
import torch.nn as nn
import utils
import numpy as np

class SphereSDF(nn.Module):
    def __init__(self, pos, r):
        super(SphereSDF, self).__init__()
        self.pos = pos
        self.r = r
    
    def forward(self, x):
        assert x.shape[-1] == self.pos.shape[-1]

        pos = utils.sync_end_dim(self.pos, x)
        x = x - pos
        s = torch.sqrt((x**2).sum(dim=-1, keepdim=True)) - self.r

        return s
        

# modified implementation
class DeepSDF(nn.Module):
    def __init__(self, num_layers):
        super(DeepSDF, self).__init__()
        self.num_layers = num_layers

        for i in range(num_layers):
            in_dim = 512
            out_dim = 512

            if i == 0:
                in_dim = 3+256  #259
            elif i == num_layers-1:
                out_dim = 1
            elif i == int(num_layers/2)-1:
                out_dim = 512 - (3+256) #259

            # weight-normalized linear FC
            setattr(self, "fc" + str(i), nn.utils.weight_norm(nn.Linear(in_dim, out_dim)))

            # weight normalization??
            setattr(self, "bn" + str(i), nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(0.1)

        for p in self.parameters():
            if len(p.shape) == 1:
                nn.init.uniform(p, 0., 0.)

    def forward(self, x):
        num_layers = self.num_layers
        x_origin = x

        for i in range(num_layers):
            fc = getattr(self, "fc" + str(i))
            bn = getattr(self, "bn" + str(i))

            if i == int(num_layers/2):
                x = torch.cat([x, x_origin], dim=-1)

            x = fc(x)

            if i < self.num_layers:
                x = F.leaky_relu(x) #torch.tanh(x)
                x = self.dropout(x)

            
        output = F.leaky_relu(x) #torch.tanh(x)
        return output


# modified implementation
class PositionalSDF(nn.Module):
    def __init__(self, num_layers):
        super(PositionalSDF, self).__init__()
        self.num_layers = num_layers
        
        # positional encoding
        self.pe = nn.Linear(3+256, 512, bias=False)
        nn.init.normal_(self.pe.weight, 0.0, 1.0)

        for i in range(num_layers):
            in_dim = 512
            out_dim = 512

            if i == 0:
                in_dim = 1024
            if i == num_layers-1:
                out_dim = 1
            elif i == int(num_layers/2)-1:
                out_dim = 512 - (3+256) #259

            # weight-normalized linear FC
            setattr(self, "fc" + str(i), nn.utils.weight_norm(nn.Linear(in_dim, out_dim)))

            # weight normalization??
            setattr(self, "bn" + str(i), nn.LayerNorm(out_dim))

        self.dropout = nn.Dropout(0.1)

        for p in self.parameters():
            if len(p.shape) == 1:
                nn.init.uniform(p, 0., 0.)

        

    def forward(self, x):
        num_layers = self.num_layers
        x_origin = x

        pi = np.pi
        x = self.pe(x)
        x = torch.cat([torch.sin(2*pi*x), torch.cos(2*pi*x)], dim=-1)

        for i in range(num_layers):
            fc = getattr(self, "fc" + str(i))
            bn = getattr(self, "bn" + str(i))

            if i == int(num_layers/2):
                x = torch.cat([x, x_origin], dim=-1)

            x = fc(x)

            if i < self.num_layers:
                x = F.leaky_relu(x) #torch.tanh(x)
                x = self.dropout(x)

            
        output = F.leaky_relu(x) #torch.tanh(x)
        return output