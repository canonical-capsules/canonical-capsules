import torch.nn.functional as F
import torch
import numpy as np
import numpy
import math

from scipy.spatial.transform import Rotation
from scipy.special import erfc, erfcinv

def transform_xyz(xyz, mat):
    # xyz: B3N, mat: B34
    ones = torch.ones(
        [xyz.shape[0], 1, xyz.shape[2]]).to(xyz.device)
    xyz = torch.cat([xyz, ones], dim=1) # B*4*N
    xyz_t = torch.matmul(mat, xyz)# B3N: Rx + t
    return xyz_t

def procruste_pose(pts0, pts1, conf=None, std_noise=1e-8):
    # pts0(Bx3xN), pts1(Bx3xN)
    indim = pts0.shape[1]
    if conf is None:
        conf = 1 / pts0.shape[-1]
    else:
        conf = (conf / conf.sum(1, keepdim=True)).unsqueeze(1) # (B1N)
    if std_noise > 0:
        pts0 = pts0 + torch.normal(0, std_noise, size=pts0.shape).to(pts0.device)
        pts1 = pts1 + torch.normal(0, std_noise, size=pts0.shape).to(pts1.device)
    center_pts0 = (pts0 * conf).sum(dim=2, keepdim=True)
    center_pts1 = (pts1 * conf).sum(dim=2, keepdim=True)

    pts0_centered = pts0 - center_pts0
    pts1_centered = pts1 - center_pts1

    cov = torch.matmul(
        pts0_centered * conf, pts1_centered.transpose(2, 1))
    
    # Faster but having memory issue. 
    # U, _, V = torch.svd(cov.cpu())
    # U = U.cuda()
    # V = V.cuda()
    # d = torch.eye(indim).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    # d[:, -1, -1] = torch.det(torch.matmul(V, U.transpose(2, 1))) # scalar
    # Vd = torch.matmul(V, d)
    # Rs = torch.matmul(Vd, U.transpose(2, 1))

    # Slower. 
    Rs = []
    for i in range(pts0.shape[0]):
        U, S, V = torch.svd(cov[i])
        d = torch.det(torch.matmul(V, U.transpose(1, 0))) # scalar
        Vd = torch.cat([V[:, :-1], V[:, -1:] * d], dim=-1)
        R = torch.matmul(Vd, U.transpose(1, 0))
        Rs += [R]
    Rs = torch.stack(Rs, dim=0)

    ts = center_pts1 - torch.matmul(Rs, center_pts0) # R * pts0 + t = pts1
    return Rs, ts


# Credit to: Daniel Rebain 
def randn_tail(shape, limits):
    comp_widths = erfc(limits / np.pi)
    widths = 1.0 - comp_widths
    inv_x = comp_widths * np.random.rand(*shape)
    inv_x = np.clip(inv_x, np.finfo(float).tiny, 2.0)
    x = erfcinv(inv_x) * np.pi
    return x

# Credit to: Daniel Rebain 
def random_rot(shape, limit=180):
    """
    Modification of the Gaussian method to limit the angle directly.
    by Daniel Rebain
    shape: (N, )
    limit: max_angle
    rot: (N, 3, 3)
    """
    limit = limit / 180.0 * np.pi

    vp = np.random.randn(*shape, 3)
    d2 = np.sum(vp**2, axis=-1)
    c2theta = np.cos(0.5 * limit)**2
    wp_limit = np.sqrt(c2theta * d2 / (1.0 - c2theta))

    comp_widths = erfc(wp_limit / np.pi)
    # widths = 1.0 - comp_widths
    inv_x = comp_widths * np.random.rand(*shape)
    inv_x = np.clip(inv_x, np.finfo(float).tiny, 2.0)
    wp = erfcinv(inv_x) * np.pi

    wp *= 2.0 * np.random.randint(2, size=shape) - 1.0
    q = np.concatenate([vp, wp[:, None]], axis=-1)
    q /= np.linalg.norm(q, axis=-1, keepdims=True)
    rot = Rotation.from_quat(q).as_matrix()
    return rot

def random_t(shape, limit=0):
    t = np.random.uniform(-1, 1, (*shape, 3, 1)) * limit 
    return t

def random_pose(N, range="uni-180-0"):
    limit_rot, limit_t = range.split("-")[1:]

    rots = random_rot((N, ), limit=float(limit_rot))
    ts = random_t((N, ), limit=float(limit_t))
    return rots, ts


def trans_pc_random(x, random_range="uni-180-0.2", return_pose="rt"):
    # x(B3N)
    indim = x.shape[1]
    if indim == 2:
        return trans_pc_random_euler(x, return_pose)
    Rs, ts = random_pose(N=len(x), range=random_range)
    Rs, ts = Rs.astype(np.float32), ts.astype(np.float32)

    Rs = torch.from_numpy(Rs).to(x.device)
    ts = torch.from_numpy(ts).to(x.device)
    x_rot = torch.matmul(Rs, x) + ts 
    # wrap qt_gt
    ts_gt = - torch.matmul(Rs.transpose(2, 1), ts) # -R't 
    Rs_gt = Rs.transpose(2, 1)
    return [Rs_gt, ts_gt], x_rot
    
def trans_pc_random_euler(x, return_pose="rt"):
    # x(B3N)
    indim = x.shape[1]
    Rs = []
    ts = []
    v_ang_x, v_ang_y, v_ang_z = 2, 2, 2
    v_t = 0.2

    for i in range(len(x)):
        anglex = np.random.uniform() * np.pi * v_ang_x
        angley = np.random.uniform() * np.pi * v_ang_y
        anglez = np.random.uniform() * np.pi * v_ang_z

        cosx = np.cos(anglex)
        cosy = np.cos(angley)
        cosz = np.cos(anglez)
        sinx = np.sin(anglex)
        siny = np.sin(angley)
        sinz = np.sin(anglez)
        if indim == 3:
            Rx = np.array([[1, 0, 0],
                            [0, cosx, -sinx],
                            [0, sinx, cosx]])
            Ry = np.array([[cosy, 0, siny],
                            [0, 1, 0],
                            [-siny, 0, cosy]])
            Rz = np.array([[cosz, -sinz, 0],
                            [sinz, cosz, 0],
                            [0, 0, 1]])
            R = Rx.dot(Ry).dot(Rz)
            Rs += [R]
            ts += [v_t * np.random.uniform(-1, 1, 3)[:, None]]
        elif indim == 2:
            # use one angle
            R = np.array([[cosx, -sinx],
                           [sinx, cosx]])
            Rs += [R]
            ts += [v_t * np.random.uniform(-1, 1, 2)[:, None]]

    Rs = np.stack(Rs).astype(np.float32)
    ts = np.stack(ts).astype(np.float32)
    Rs = torch.from_numpy(Rs).to(x.device)
    ts = torch.from_numpy(ts).to(x.device)
    x_rot = torch.matmul(Rs, x) + ts 
    # wrap qt_gt
    ts_gt = - torch.matmul(Rs.transpose(2, 1), ts) # -R't 
    Rs_gt = Rs.transpose(2, 1)
    return [Rs_gt, ts_gt], x_rot


def get_rots(indim, bin_size=0.05):
    degsx = np.arange(0, 1, bin_size) * np.pi * 2 
    degsy = np.arange(0, 1, bin_size) * np.pi * 2 
    degsz = np.arange(0, 1, bin_size) * np.pi * 2 
    Rs = []

    for anglex, angley, anglez in zip(degsx, degsy, degsz):
        if indim == 2:
            cosx = np.cos(anglex)
            sinx = np.sin(anglex)
            R = np.array([[cosx, -sinx],
                           [sinx, cosx]]).astype(np.float32)
        else:
            cosx = np.cos(anglex)
            cosy = np.cos(angley)
            cosz = np.cos(anglez)
            sinx = np.sin(anglex)
            siny = np.sin(angley)
            sinz = np.sin(anglez)
            Rx = np.array([[1, 0, 0],
                            [0, cosx, -sinx],
                            [0, sinx, cosx]])
            Ry = np.array([[cosy, 0, siny],
                            [0, 1, 0],
                            [-siny, 0, cosy]])
            Rz = np.array([[cosz, -sinz, 0],
                            [sinz, cosz, 0],
                            [0, 0, 1]])
            R = Rx.dot(Ry).dot(Rz).astype(np.float32)

        Rs += [R[None]]
    return Rs
    
#
# geom_torch.py ends here