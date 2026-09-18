import torch
import torch.nn as nn




def projection(grid, w2cs, ixts):
    dtype = w2cs.dtype
    if grid.dtype != dtype: grid = grid.to(dtype)
    if ixts.dtype != dtype: ixts = ixts.to(dtype)
    points = grid.reshape(1,-1,3) @ w2cs[:,:3,:3].permute(0,2,1) + w2cs[:,:3,3][:,None]
    points = points @ ixts.permute(0,2,1)
    points_xy = points[...,:2]/points[...,-1:]
    return points_xy, points[...,-1:]


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

def conv_nd(dims, *args, **kwargs):
    """
    Create a 1D, 2D, or 3D convolution module.
    """
    if dims == 1:
        return nn.Conv1d(*args, **kwargs)
    elif dims == 2:
        return nn.Conv2d(*args, **kwargs)
    elif dims == 3:
        return nn.Conv3d(*args, **kwargs)
    raise ValueError(f"unsupported dimensions: {dims}")

def Normalize(in_channels, norm_type='group'):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)

def AdaptiveNormalize(in_channels, norm_type='group', affine=False):
    assert norm_type in ['group', 'batch']
    if norm_type == 'group':
        return nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=affine)
    elif norm_type == 'batch':
        return nn.SyncBatchNorm(in_channels)
    
def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


