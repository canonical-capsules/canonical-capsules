import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F


def evaluate_pose(x, att):
    # x: B3N, att: B1KN1
    # ts: B3k1
    pai = att.sum(dim=3, keepdim=True) # B1K11
    att = att / torch.clamp(pai, min=1e-3)

    ts = torch.sum(
        att * x[:, :, None, :, None], dim=3) # B3K1
    return ts


def spatial_variance(x, att, norm_type="l2"):
    pai = att.sum(dim=3, keepdim=True) # B1K11
    att = att / torch.clamp(pai, min=1e-3)
    ts = torch.sum(
        att * x[:, :, None, :, None], dim=3) # B3K1

    x_centered = x[:, :, None] - ts # B3KN
    x_centered = x_centered.permute(0, 2, 3, 1) # BKN3
    att = att.squeeze(1) # BKN1
    cov = torch.matmul(
        x_centered.transpose(3, 2), att * x_centered) # BK33
    
    # l2 norm
    vol = torch.diagonal(cov, dim1=-2, dim2=-1).sum(2) # BK
    if norm_type == "l2":
        vol = vol.norm(dim=1).mean()
    elif norm_type == "l1":
        vol = vol.sum(dim=1).mean()
    else:
        # vol, _ = torch.diagonal(cov, dim1=-2, dim2=-1).sum(2).max(dim=1)
        raise NotImplementedError
    return vol


def reg_att(att, x, config, **kwargs):

    loss_dict = {}

    # Localization loss 
    if config.loss_volume > 0:
        loss_volume = spatial_variance(
            x, att, norm_type=config.spatial_var_norm)
        loss_dict["loss_volume"] = loss_volume * config.loss_volume

    # Equilibrium loss 
    if config.loss_att_amount > 0:
        pai = att.sum(dim=3, keepdim=True) # B1K11
        loss_att_amount = torch.var(pai.reshape(pai.shape[0], -1), dim=1).mean()
        loss_dict["loss_att_amount"] = loss_att_amount * config.loss_att_amount

    # Equivariance loss 
    if config.loss_2cps > 0:
        rt_rels = kwargs["rt_rels"] # gt relative pose among two views
        # get kps
        if not "kps" in kwargs:
            if config.att_type_out in ["gmm", "None"]:
                pai = att.sum(dim=3, keepdim=True) # B1K11
                att = att / torch.clamp(pai, min=1e-3)

            kps = torch.sum(
                att * x[:, :, None, :, None], dim=3).squeeze(-1)# B3K
            assert kps.shape[0] % config.num_view == 0
            bs = kps.shape[0] // config.num_view
            kps0 = kps[:bs]
            kps1s = [kps[bs*i:bs*(i+1)] for i in range(1, config.num_view)]
        else:
            kps0, kps1s = kwargs["kps"]

        loss_2cps = []
        for i in range(config.num_view - 1):
            rt_rel = rt_rels[i]
            R_gt, T_gt = rt_rel
            kps0_can = torch.matmul(R_gt, kps0) + T_gt
            kps1 = kps1s[i]
            loss_2cps += [((kps0_can - kps1) ** 2).sum(1).mean()]
        loss_2cps = torch.stack(loss_2cps).mean()
        loss_dict["loss_2cps"] = loss_2cps * config.loss_2cps
    
    # to sum up losses
    loss_dict["sum"] = torch.stack(list(loss_dict.values())).sum()    
    return loss_dict
    
    