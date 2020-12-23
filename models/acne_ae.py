import os
import torch
import copy
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from model_utils.decode_utils import ChamferLoss, KpDecoder, MlpPointsFC
from acne_utils import AcneKpEncoder
from geom_torch import trans_pc_random, procruste_pose, get_rots 
from loss_util import *
from vis_util import *


class AcneAe(nn.Module):
    def __init__(self, config):
        super(AcneAe, self).__init__()
        self.config = config
        encoder = eval(self.config.feat_net)
        decoder = eval(self.config.ae_decoder)

        # size of each capsule's descriptor
        gc_dim = config.acne_dim 
        indim = config.indim
        self.encoder = encoder(config, indim)

        self.aligner = None
        if config.aligner == "pretrained":
            # Use pretrained aligner (trained separately)
            pass
        elif config.aligner == "init":
            # init, you have to train it.
            aligner_config = copy.copy(config)
            aligner_config.aligner = "None" 
            aligner_config.ref_kp_type = "mlp_fc"
            aligner_config.loss_reg_att_f = 1 # localization/equi/equili losses 
            aligner_config.loss_beta = 1 # invariance loss  
            aligner_config.loss_ref_kp_can = 1 # canonical loss
            self.aligner = AcneAeAligner(aligner_config)
        elif config.aligner == "None":
            pass
        else:
            raise ValueError("no such aligner")
        # TODO: don't need to initiate decoder for aligner
        self.decoder = decoder(
            self.config.acne_num_g, gc_dim,
            self.config.num_pts, self.config)


        self.chamfer_loss = ChamferLoss()
        # Keypiont Regressor: 
        self.ref_kp_type = config.ref_kp_type
        if config.ref_kp_type == "mlp_fc":
            self.ref_kp_net = MlpPointsFC(
                gc_dim*config.acne_num_g, config.acne_num_g, indim, config)

    def forward_test(self, in_dict, vis_fn=None):
        data = in_dict["data"]
        mode = in_dict["mode"]
        writer = in_dict["writer"]
        iter_idx = in_dict["iter_idx"]
        pc = data["pc"]

        rt = None
        if "transform" in data:
            # fixed data during testing
            rt = [data["transform"][:, :3, :3], data["transform"][:, :3, 3:4]]
        att_aligner = None
        if self.aligner is not None:
            self.aligner.eval()
            with torch.no_grad():
                pc, ret_aligner = self.aligner.forward_align(pc, rt=rt)
                att_aligner = ret_aligner["att_aligner"]

        x = pc.transpose(2, 1)

        # encoding x
        input_feat = x[..., None]
        gc_att = self.encoder(input_feat, att_aligner=att_aligner, return_att=True) # BCK1, B1GN1
        gc, att = gc_att

        # Evaluating pose 
        pose_local = evaluate_pose(x, att)
        kps = pose_local.squeeze(-1)

        # w/o canonicalized descriptor
        if self.config.pose_block == "procruste":
            if self.ref_kp_type != "None":
                if self.ref_kp_type.startswith("mlp"):
                    kps_ref= self.ref_kp_net(gc.reshape(gc.shape[0], -1))
                    kps_ref = kps_ref - kps_ref.mean(dim=2, keepdim=True)
                else:
                    raise NotImplementedError
                R_can, T_can = procruste_pose(kps, kps_ref, std_noise=0) # kps_ref = R * kpsi  + T
                kps = torch.matmul(R_can, kps) + T_can
            else:
                R_can = None

        # reconstruction from canonical capsules 
        gc = torch.cat([kps[..., None], gc], dim=1)
        y = self.decoder(gc.transpose(2, 1).squeeze(-1))
        acc = {}
        acc["chamfer_error"] = self.chamfer_loss(pc, y)
        return acc, {} 

    def forward(self, in_dict, vis_fn=None):
        data = in_dict["data"]
        mode = in_dict["mode"]
        writer = in_dict["writer"]
        iter_idx = in_dict["iter_idx"]
        # Test here
        if mode in ["test", "valid"]:
            return self.forward_test(in_dict, vis_fn=None)
        
        pc = data["pc"]

        att_aligner = None
        if self.aligner is not None:
            if self.config.loss_aligner > 0:
                self.aligner.train()
                pc, ret_aligner = self.aligner.forward_align(pc, mode="train", iter_idx=iter_idx, writer=writer)
                att_aligner = ret_aligner["att_aligner"] 
                loss_aligner = ret_aligner["loss"] 
            else:
                # For pretrained aligner
                self.aligner.eval()
                with torch.no_grad():
                    pc, ret_aligner = self.aligner.forward_align(pc, mode="test")
                    att_aligner = ret_aligner["att_aligner"]

        x = pc.transpose(2, 1)
        if self.config.pose_block == "procruste":
            # w/o canical descriptor
            bs = x.shape[0]
            rt_0, x_rot0 = trans_pc_random(x, random_range=self.config.random_range, return_pose="rt")
            rts, x_rots, rt_rels = [], [], []
            for i in range(1, self.config.num_view):
                rt_1, x_rot1 = trans_pc_random(x, random_range=self.config.random_range, return_pose="rt")
                rts += [rt_1]
                x_rots += [x_rot1]
                rt_rels += [[
                    torch.matmul(rt_1[0].transpose(2, 1), rt_0[0]),
                    torch.matmul(rt_1[0].transpose(2, 1), rt_0[1] - rt_1[1])
                    ]] # x1 = rt_rel[0] * x0 + rt_rel[1] 
            x = torch.cat([x_rot0] + x_rots, dim=0)
            if self.ref_pcd is not None:
                x = torch.cat([self.ref_pcd, x], dim=0)

        input_feat = x[..., None]
        gc_att = self.encoder(input_feat, att_aligner=att_aligner, return_att=True) # BCK1, B1GN1
        gc, att = gc_att

        # Evaluating pose 
        pose_local = evaluate_pose(x, att)
        kps = pose_local.squeeze(-1)

        # solve relative pose and transform
        if self.config.pose_block == "procruste":
            # w/o canical descriptor
            kps0 = pose_local[:bs].squeeze(-1) # B3K
            Rs, Ts, kps1s = [], [], []
            Rs_inv, Ts_inv = [], []
            for i in range(1, self.config.num_view):
                kps1 = pose_local[bs*i:bs*(i+1)].squeeze(-1) # B3K
                kps1s += [kps1]
                noise = self.config.procruste_noise if self.training else 0
                    
            # prepare input of decoder and the ground-truth 
            if self.ref_kp_type != "None":
                kps_source = pose_local[self.num_ref_pcd:].squeeze(-1)
                if self.ref_kp_type.startswith("mlp"):
                    kps_ref= self.ref_kp_net(gc.reshape(gc.shape[0], -1))
                    kps_ref = kps_ref - kps_ref.mean(dim=2, keepdim=True)
                else:
                    raise NotImplementedError
                # kps_can: canonicalized pose
                noise = self.config.procruste_noise if self.training else 0
                R_can, T_can = procruste_pose(kps_source, kps_ref, std_noise=noise) # kps_ref = R * kpsi  + T
                kps_can = torch.matmul(R_can, kps_source) + T_can
                pc_can = torch.matmul(R_can, x[self.num_ref_pcd:]) + T_can
                pc_can = pc_can.transpose(2, 1)
            else:
                # No Canonicalizer
                kps_source = pose_local[self.num_ref_pcd:].squeeze(-1)
                kps_can = kps_source
                pc_can = x.transpose(2, 1)
            gc = torch.cat([kps_can[..., None], gc[self.num_ref_pcd:]], dim=1)
        else:
            gc = torch.cat([pose_local, gc], dim=1)

        # losses
        loss = 0
        report = iter_idx % self.config.rep_intv == 0
        if self.config.loss_reconstruction > 0:
            y = self.decoder(gc.transpose(2, 1).squeeze(-1))
            if self.config.pose_block == "procruste":
                loss_recon = self.chamfer_loss(pc_can, y)
            else:
                loss_recon = self.chamfer_loss(pc, y)
            if report:
                writer.add_scalar(
                    "loss_recon", loss_recon, global_step=iter_idx)
            loss += self.config.loss_reconstruction * loss_recon

        if self.config.loss_aligner > 0:
            if report:
                writer.add_scalar(
                    "loss_aligner", loss_aligner, global_step=iter_idx)
            loss += self.config.loss_aligner * loss_aligner

        # w\o canonicalizing descriptors
        if self.config.loss_reg_att_f > 0:
            reg_att_f_dict = reg_att(att, x, self.config, rt_rels=rt_rels, kps=[kps0, kps1s])
            if report:
                for item in reg_att_f_dict.keys():
                    writer.add_scalar(
                        f"loss_reg_att_f/{item}",
                        reg_att_f_dict[item], global_step=iter_idx)
            loss += self.config.loss_reg_att_f * reg_att_f_dict["sum"]

        if self.config.loss_beta > 0:
            # consider two views
            beta = gc[:, 3:]
            beta0 = beta[:bs] 
            beta1 = beta[bs:]
            loss_beta = F.mse_loss(beta0, beta1)
            if report:
                writer.add_scalar(
                    "loss_beta", loss_beta, global_step=iter_idx)
            loss += self.config.loss_beta * loss_beta 

        if self.config.loss_ref_kp_can > 0:
            # Sync
            loss_ref_kp_can = ((kps_can - kps_ref) ** 2).sum(1).mean()
            loss += self.config.loss_ref_kp_can * loss_ref_kp_can  
            if report:
                writer.add_scalar(
                    "loss_ref_kp_can", loss_ref_kp_can , global_step=iter_idx)

        return loss
    
    def vis_single(self, in_dict, vis_dump_dir):
        # rotate input and show the decomposition and reconstruction 
        data = in_dict["data"]
        mode = in_dict["mode"]
        writer = in_dict["writer"]
        iter_idx = in_dict["iter_idx"]
        pc = data["pc"]

        assert pc.shape[2] == self.config.indim
        x_can = pc.transpose(2, 1)
        idx = 0
        Rs = get_rots(self.config.indim)

        # png
        vis_fn_decomp = os.path.join(vis_dump_dir, "decomp")
        vis_fn_recon = os.path.join(vis_dump_dir, "recon")

        for R in Rs:
            idx += 1
            x = torch.matmul(
                torch.from_numpy(R).to(x_can.device), x_can)
            
            att_aligner = None
            if self.aligner is not None:
                with torch.no_grad():
                    self.aligner.eval()
                    x_, ret_aligner = self.aligner.forward_align(x.transpose(2, 1), mode="vis")
                    att_aligner = ret_aligner["att_aligner"]

                    # Vis decomposition 
                    if not os.path.exists(vis_fn_decomp):
                        os.makedirs(vis_fn_decomp)
                    vis_fn = os.path.join(vis_fn_decomp, f"{idx}".zfill(3)) 
                    vis_pts(x, att_aligner, vis_fn=vis_fn)

                    x = x_.transpose(2, 1)

            input_feat = x[..., None]
            gc_att = self.encoder(input_feat, att_aligner=att_aligner, return_att=True) # BCK1, B1GN1
            gc, att = gc_att

            # Evaluating pose 
            pose_local = evaluate_pose(x, att)
            kps = pose_local.squeeze(-1)
            gc = torch.cat([kps[..., None], gc], dim=1)
            pc_recons = self.decoder(
                gc.transpose(2, 1).squeeze(-1), return_splits=True)
            y = torch.cat(pc_recons, dim=2).transpose(2, 1)
            loss_chamfer = self.chamfer_loss(x.transpose(2, 1), y)
            print(f"R: {R}; loss: {loss_chamfer}")

            # Reconstruction in cannical pose
            if not os.path.exists(vis_fn_recon):
                os.makedirs(vis_fn_recon)
            vis_fn = os.path.join(vis_fn_recon, f"{idx}".zfill(3)) 
            vis_recon(pc_recons, vis_fn)
        
        # generate gif
        gif_fns = [vis_fn_decomp, vis_fn_recon]
        
        for gif_fn in gif_fns:
            png_fn = os.path.join(gif_fn, "*.png")
            animation_fn = os.path.join(gif_fn, "animation.gif")
            cmd = "convert -delay 10 -loop 0 {} {}".format(png_fn, animation_fn)
            os.system(cmd)
            cmd = "rm {}".format(png_fn)
            os.system(cmd)


class AcneAeAligner(AcneAe):
    def forward_align(self, pc, rt=None, mode="test", iter_idx=None, writer=None):
        # BN3
        x = pc.transpose(2, 1)
        bs = x.shape[0]

        rt_rels = None
        if self.config.mode != "vis" and self.config.num_view > 1:
            # num_view = 1 means we nevect enforce transformation invariance.
            if mode == "test":
                if rt is None:
                    rt, x = trans_pc_random(x, random_range=self.config.random_range, return_pose="rt")
                else:
                    # Fixed rt during testing
                    x = torch.matmul(rt[0], x) + rt[1] 
            elif mode == "train":
                bs = x.shape[0]
                rt_0, x_rot0 = trans_pc_random(x, random_range=self.config.random_range, return_pose="rt")
                rts, x_rots, rt_rels = [], [], []
                for i in range(1, self.config.num_view):
                    rt_1, x_rot1 = trans_pc_random(x, random_range=self.config.random_range, return_pose="rt")
                    rts += [rt_1]
                    x_rots += [x_rot1]
                    rt_rels += [[
                        torch.matmul(rt_1[0].transpose(2, 1), rt_0[0]),
                        torch.matmul(rt_1[0].transpose(2, 1), rt_0[1] - rt_1[1])
                        ]] # x1 = rt_rel[0] * x0 + rt_rel[1] 
                x = torch.cat([x_rot0] + x_rots, dim=0)

        # update pc
        input_feat = x[..., None]
        gc_att = self.encoder(input_feat, return_att=True) # BCK1, B1GN1
        gc, att = gc_att

        # Evaluating pose 
        pose_local = evaluate_pose(x, att)
        kps = pose_local.squeeze(-1)

        # Alignment
        kps_ref= self.ref_kp_net(gc.reshape(gc.shape[0], -1))
        kps_ref = kps_ref - kps_ref.mean(dim=2, keepdim=True)
        R_can, T_can = procruste_pose(kps, kps_ref, std_noise=0) # kps_ref = R * kpsi  + T

        x_can = torch.matmul(R_can, x) + T_can
        kps_can = torch.matmul(R_can, kps) + T_can

        ret_dict = {}
        if mode == "train":
            kps0 = pose_local[:bs].squeeze(-1) # B3K
            Rs, Ts, kps1s = [], [], []
            for i in range(1, self.config.num_view):
                kps1 = pose_local[bs*i:bs*(i+1)].squeeze(-1) # B3K
                kps1s += [kps1]

            loss = 0
            report = iter_idx % self.config.rep_intv == 0

            if self.config.loss_reg_att_f > 0:
                reg_att_f_dict = reg_att(att, x, self.config, rt_rels=rt_rels, kps=[kps0, kps1s])
                if report:
                    for item in reg_att_f_dict.keys():
                        writer.add_scalar(
                            f"aligner/loss_reg_att_f/{item}",
                            reg_att_f_dict[item], global_step=iter_idx)
                loss += self.config.loss_reg_att_f * reg_att_f_dict["sum"]

            if self.config.loss_beta > 0:
                # consider two views
                beta = gc
                beta0 = beta[:bs] 
                beta1 = beta[bs:]
                loss_beta = F.mse_loss(beta0, beta1)
                if report:
                    writer.add_scalar(
                        "aligner/loss_beta", loss_beta, global_step=iter_idx)
                loss += self.config.loss_beta * loss_beta 
            
            if self.config.loss_ref_kp_can > 0:
                # Sync
                loss_ref_kp_can = ((kps_can - kps_ref) ** 2).sum(1).mean()
                loss += self.config.loss_ref_kp_can * loss_ref_kp_can  
                if report:
                    writer.add_scalar(
                        "aligner/loss_ref_kp_can", loss_ref_kp_can , global_step=iter_idx)
            ret_dict["att_aligner"] = att[:bs]
            ret_dict["kps_ref"] = kps_ref[:bs]
            ret_dict["loss"] = loss
            
            return x_can[:bs].transpose(2, 1), ret_dict 
        else:
            ret_dict["att_aligner"] = att
            ret_dict["kps_ref"] = kps_ref
            ret_dict["kps"] = kps
            ret_dict["rt_can"] = [R_can, T_can] 
            return x_can.transpose(2, 1), ret_dict 
#
# acne_ae.py ends here 