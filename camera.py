import torch
import torch.nn.functional as F
import torch.nn as nn

def decode_pose(pose):
    # (..., 5) -> (..., 4, 4)
    decoded_pose = torch.zeros((*pose.shape[:-1], 4, 4), device=pose.device)

    decoded_pose[..., 0, 0]
    decoded_pose[..., 0, 0]
    decoded_pose[..., 0, 0]
    decoded_pose[..., 0, 0]
    decoded_pose[..., 0, 0]
    decoded_pose[..., 0, 0]
    decoded_pose[..., 0, 0]

    return decoded_pose

def encode_pose(pose):
    # (..., 4, 4) -> (..., 5)
    pass

# uv to world
class OrthogonalCamera(nn.Module):
    def __init__(self):
        super(OrthogonalCamera, self).__init__()
    
    def forward(self, uv, depth, pose):
        # uv : (..., pose_batch, N, 2)
        # depth : (..., pose_batch, N, 1)
        # pose : (..., pose_batch, 5)

        assert depth.shape[-1] == 1

        uvd = torch.cat([uv, depth], dim=-1)

        decoded_pose = decode_pose(pose)        # pose : (..., pose_batch, 4, 4)
        decoded_pose = decoded_pose.unsqueeze(-3)
        uvd1 = torch.cat([uvd, torch.ones_like(depth)], dim=-1)
        world_pos = torch.matmul(uvd1, decoded_pose.transpose(-1,-2))

        return world_pos


class PerspectiveCamera(nn.Module):
    def __init__(self):
        super(PerspectiveCamera, self).__init__()
    
    def forward(self, uv, depth, pose):
        # uv : (..., pose_batch, N, 2)
        # depth : (..., pose_batch, N, 1)
        # pose : (..., pose_batch, 5)

        uv1 = torch.cat([uv, torch.ones_like(depth)], dim=-1)
        duv1 = depth*uv1

        decoded_pose = decode_pose(pose)
        decoded_pose = decoded_pose.unsqueeze(-3)
        duv11 = torch.cat([duv1, torch.ones_like(depth)], dim=-1)
        world_pos = torch.matmul(duv11, decoded_pose.transpose(-1,-2))

        return world_pos


# world to uv
class OrthogonalProject(nn.Module):
    def __init__(self):
        super(OrthogonalProject, self).__init__()
    
    def forward(self, x, pose):
        # x : (..., pose_batch, N, 3)
        # pose : (..., pose_batch, 5)

        decoded_pose = decode_pose(pose)
        decoded_pose = decoded_pose.unsqueeze(-3)
        decoded_pose = torch.inverse(decoded_pose)

        cam_pos = torch.matmul(x, decoded_pose.transpose(-1,-2))
        cam_pos = cam_pos[..., :2]

        return cam_pos

class PerspectiveProject(nn.Module):
    def __init__(self):
        super(PerspectiveProject, self).__init__()
    
    def forward(self, x, pose):
        # x : (..., pose_batch, N, 3)
        # pose : (..., pose_batch, 5)
        
        decoded_pose = decode_pose(pose)
        decoded_pose = decoded_pose.unsqueeze(-3)
        decoded_pose = torch.inverse(decoded_pose)

        cam_pos = torch.matmul(x, decoded_pose.transpose(-1,-2))
        cam_pos = cam_pos[..., :2] / cam_pos[..., -2:-1]

        return cam_pos