import matplotlib.pyplot as plt
import torch
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os


def vis_pts(x, att=None, vis_fn="temp.png"):
    idx = 0
    pts = x[idx].transpose(1, 0).cpu().numpy()
    label_map = None
    if att is not None:
        label_map = torch.argmax(att, dim=2)[idx].squeeze().cpu().numpy()
    vis_pts_att(pts, label_map, fn=vis_fn)
    

def vis_recon(pc_recon, vis_fn="temp.png", idx=0):
    # pc_recon: a list of sets of points
    # idx = 0
    label_map = []
    pts = []
    for i, patch in enumerate(pc_recon):
        pts_cur = patch[idx].transpose(1, 0).cpu().numpy()
        label_map += [np.ones(len(pts_cur)) * i]
        pts += [pts_cur]
    
    pts = np.concatenate(pts, axis=0)
    label_map = np.concatenate(label_map, axis=0)
    vis_pts_att(pts, label_map, fn=vis_fn)


def vis_pts_att(pts, label_map, fn="temp.png", marker=".", alpha=0.9):
    # pts (n, d): numpy, d-dim point cloud
    # label_map (n, ): numpy or None
    # fn: filename of visualization
    assert pts.shape[1] in [2, 3]
    if pts.shape[1] == 2:
        xs = pts[:, 0]
        ys = pts[:, 1]
        if label_map is not None:
            plt.scatter(xs, ys, c=label_map, cmap="jet", marker=".", alpha=0.9, edgecolor="none")
        else:
            plt.scatter(xs, ys, c="grey", alpha=0.8, edgecolor="none")
        # save
        plt.xlim(-1,1)
        plt.ylim(-1,1)
        plt.axis("off")
    elif pts.shape[1] == 3:
        TH = 0.7
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.set_zlim(-TH,TH)
        ax.set_xlim(-TH,TH)
        ax.set_ylim(-TH,TH)
        xs = pts[:, 0]
        ys = pts[:, 1]
        zs = pts[:, 2]
        if label_map is not None:
            ax.scatter(xs, ys, zs, c=label_map, cmap="jet", marker=marker, alpha=alpha)
        else:
            ax.scatter(xs, ys, zs, marker=marker, alpha=alpha, edgecolor="none")

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,)
    plt.close()