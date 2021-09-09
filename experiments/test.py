import torch
import torch.nn.functional as F
import torch.nn as nn

import sampler
import camera
import optimization
import analytic_sdf

if __name__ == "__main__":
    # create modules
    cam = camera.PerspectiveCamera()
    sdf = analytic_sdf.SphereSDF(torch.zeros((3)), 1)

    # create samples
    uv = sampler.grid_2d(64, 64).requires_grad_(True)
    depth = torch.zeros((*uv.shape[:-1], 1), device=uv.device)
    pose = torch.zeros((4,4), device=uv.device)
    pose[:2, :2] = torch.eye(3)
    pose[2,3] = -2

    foo = lambda d: sdf(cam(uv, d, pose))
    
    for i in range(5):
        depth = optimization.lm(depth, foo)

    
