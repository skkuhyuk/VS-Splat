import torch, math
import numpy as np
from sklearn.cluster import KMeans
import re

def build_rays_torch(c2ws, ixts, H, W, scale=1.0):

    H, W = int(H*scale), int(W*scale)
    ixts[:,:2] *= scale
    rays_o = c2ws[:,:3, 3][:,None,None]
    X, Y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    XYZ = torch.cat((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, torch.ones_like(X[:, :, None])), dim=-1).to(c2ws)
    
    i2ws = torch.inverse(ixts).permute(0,2,1) @ c2ws[:,:3, :3].permute(0,2,1)
    XYZ = torch.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(1,H,1,1)
    rays_o = rays_o.repeat(1,1,W,1)
    rays = torch.cat((rays_o, XYZ), dim=-1)
    return rays

def build_rays(c2ws, ixts, H, W, scale=1.0):

    H, W = int(H*scale), int(W*scale)
    ixts[:,:2] *= scale

    rays_o = c2ws[:,:3, 3][:,None,None]
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    XYZ = np.concatenate((X[:, :, None] + 0.5, Y[:, :, None] + 0.5, np.ones_like(X[:, :, None])), axis=-1)
    i2ws = np.linalg.inv(ixts).transpose(0,2,1) @ c2ws[:,:3, :3].transpose(0,2,1)
    XYZ = np.stack([(XYZ @ i2w) for i2w in i2ws])
    rays_o = rays_o.repeat(H, axis=1)
    rays_o = rays_o.repeat(W, axis=2)
    rays = np.concatenate((rays_o, XYZ), axis=-1)
    return rays.astype(np.float32)

def build_rays_ortho(c2ws, H, W, scale=1.0):

    c2ws_rot = c2ws[:,:3,:3]
    c2ws_t = c2ws[:,:3,3].reshape(-1,1,3)
            
    rays_d = torch.zeros(1,1,3).to(c2ws)
    rays_d[...,-1] = 1.0
    rays_d = rays_d @ c2ws_rot.transpose(1,2)
    rays_d = rays_d[:,None].expand(-1,H,W,-1)
            
    X, Y = np.meshgrid(np.arange(W), np.arange(H))
    X = torch.from_numpy(X[:, :, None] + 0.5).float()/W * 2 - 1.0
    Y = torch.from_numpy(Y[:, :, None] + 0.5).float()/H * 2 - 1.0
    XYZ = torch.cat((X*scale, Y*scale, torch.zeros_like(X)), dim=-1).to(c2ws)
    XYZ = XYZ.view(1,-1,3)
    rays_o = XYZ @ c2ws_rot.transpose(1,2) + c2ws_t
    rays = torch.cat((rays_o.view(rays_d.shape), rays_d), dim=-1)
    return rays

def KMean(xyz, n_clusters):
    kmeans = KMeans(n_clusters = n_clusters, n_init=10, random_state=20211202)
    kmeans.fit(xyz)
    labels = kmeans.labels_
    
    clusters = []
    for i in range(n_clusters):
        idx = np.where(labels==i)[0]
        clusters.append(idx.astype(np.uint8))

    return clusters

def fov_to_ixt(fov, reso):
    ixt = np.eye(3, dtype=np.float32)
    ixt[0][2], ixt[1][2] = reso[0]/2, reso[1]/2
    focal = .5 * reso / np.tan(.5 * fov)
    ixt[[0,1],[0,1]] = focal
    return ixt

def intrinsic_to_fov(K, w=None, h=None):
    # Extract the focal lengths from the intrinsic matrix
    fx = K[0, 0]
    fy = K[1, 1]
    
    w = K[0, 2]*2 if w is None else w
    h = K[1, 2]*2 if h is None else h
    
    # Calculating field of view
    fov_x = 2 * np.arctan2(w, 2 * fx) 
    fov_y = 2 * np.arctan2(h, 2 * fy)
    
    return fov_x, fov_y

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def pose_sub_selete(poses, N_poses):
    # to select N_poses poses that close to the equatorial
    # Define the camera's local up-vector and the world's up-vector
    camera_up_vector = np.array([0, 1, 0])
    world_up_vector = np.array([1, 0, 0])

    # Extract the rotation matrices from the transformation matrices
    rotations = poses[:, :3, :3]  # Shape: [num_poses, 3, 3]

    # Transform the camera's up-vector to world coordinates for all poses
    camera_up_world = np.einsum('ijk,k->ij', rotations, camera_up_vector)  # Shape: [num_poses, 3]

    # Normalize the transformed up-vectors
    camera_up_world_norm = camera_up_world / np.linalg.norm(camera_up_world, axis=1)[:, np.newaxis]

    # Calculate the dot product with the world up-vector to find cosines of angles
    cos_angles = np.dot(camera_up_world_norm, world_up_vector)

    # Calculate angles in radians using arccos
    angles = np.arccos(np.clip(cos_angles, -1.0, 1.0))

    # Select the poses with the smallest angles (closest to the equatorial plane)
    indices = np.argsort(angles)[:N_poses]
    
    return indices

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

# Matrix to quaternion does not come under NVIDIA Copyright
# Written by Stan Szymanowicz 2023
def matrix_to_quaternion(M: torch.Tensor) -> torch.Tensor:
    """
    Matrix-to-quaternion conversion method. Equation taken from 
    https://www.euclideanspace.com/maths/geometry/rotations/conversions/matrixToQuaternion/index.htm
    Args:
        M: rotation matrices, (3 x 3)
    Returns:
        q: quaternion of shape (4)
    """
    tr = 1 + M[ 0, 0] + M[ 1, 1] + M[ 2, 2]

    if tr > 0:
        r = torch.sqrt(tr) / 2.0
        x = ( M[ 2, 1] - M[ 1, 2] ) / ( 4 * r )
        y = ( M[ 0, 2] - M[ 2, 0] ) / ( 4 * r )
        z = ( M[ 1, 0] - M[ 0, 1] ) / ( 4 * r )
    elif ( M[ 0, 0] > M[ 1, 1]) and (M[ 0, 0] > M[ 2, 2]):
        S = torch.sqrt(1.0 + M[ 0, 0] - M[ 1, 1] - M[ 2, 2]) * 2 # S=4*qx 
        r = (M[ 2, 1] - M[ 1, 2]) / S
        x = 0.25 * S
        y = (M[ 0, 1] + M[ 1, 0]) / S 
        z = (M[ 0, 2] + M[ 2, 0]) / S 
    elif M[ 1, 1] > M[ 2, 2]: 
        S = torch.sqrt(1.0 + M[ 1, 1] - M[ 0, 0] - M[ 2, 2]) * 2 # S=4*qy
        r = (M[ 0, 2] - M[ 2, 0]) / S
        x = (M[ 0, 1] + M[ 1, 0]) / S
        y = 0.25 * S
        z = (M[ 1, 2] + M[ 2, 1]) / S
    else:
        S = torch.sqrt(1.0 + M[ 2, 2] - M[ 0, 0] -  M[ 1, 1]) * 2 # S=4*qz
        r = (M[ 1, 0] - M[ 0, 1]) / S
        x = (M[ 0, 2] + M[ 2, 0]) / S
        y = (M[ 1, 2] + M[ 2, 1]) / S
        z = 0.25 * S

    return torch.stack([r, x, y, z], dim=-1)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getView2World(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = C2W
    return np.float32(Rt)




