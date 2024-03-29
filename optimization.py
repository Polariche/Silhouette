import torch
import torch.nn.functional as F
import torch.nn as nn

def gauss_newton(x, f):
    y = f(x)
    dx = torch.autograd.grad(y, [x], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
    dx_pinv = torch.pinverse(dx.unsqueeze(-2))[..., 0]
 
    print(dx_pinv)
 
    return x - y*dx_pinv
 
 
def lm(x, f, lamb = 1.1):
    y = f(x)
    dx = torch.autograd.grad(y, [x], grad_outputs=torch.ones_like(y), retain_graph=True, create_graph=True)[0]
 
    J = dx.unsqueeze(-1)
    Jt = J.transpose(-2, -1)
    JtJ = torch.matmul(Jt, J)

    k = JtJ.shape[-1]
 
    diag_JtJ = torch.cat([JtJ[..., i, i] for i in range(k)])
    diag_JtJ = diag_JtJ.view(-1, k, 1)
    diag_JtJ = torch.eye(k, device=x.device).unsqueeze(0).expand(diag_JtJ.shape[0], -1, -1) * diag_JtJ
 
    pinv = torch.matmul(torch.inverse(JtJ + lamb * diag_JtJ), Jt)

    delta = - pinv * y.unsqueeze(-1)
    delta = delta[..., 0, :]
 
    return x + delta