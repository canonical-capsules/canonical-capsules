import sys
import os
import time
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
import pickle
import h5py

from util import log


class Network(object):
    """Wrapper for training and testing procedure"""

    def __init__(self, config):
        """init"""
        self.config = config
        # make network deterministic
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        np.random.seed(1234)

        # build model
        from models.acne_ae import AcneAe 
        # from models.acne import Acne
        model = eval(self.config.model)
        model = model(config) # 2_1 output one_dimensional score

        if self.config.use_cuda:
            model.cuda()
        self.model = model

        # build optimizer
        optimizer = optim.Adam(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        # setup scheduler
        if self.config.scheduler == 1:
            # For the unaligned dataset
            self.config.num_epoch = 450
            self.scheduler = MultiStepLR(
                optimizer, milestones=[200, 400], gamma=0.1)
        elif self.config.scheduler == 2:
            # For the aligned dataset
            self.config.num_epoch = 325 # same as AtlasNet
            self.scheduler = MultiStepLR(
                optimizer, milestones=[250, 300], gamma=0.1)
        elif self.config.scheduler == 3:
            # For the aligned dataset
            self.config.num_epoch = 250 # same as AtlasNet
            self.scheduler = MultiStepLR(
                optimizer, milestones=[150, 200], gamma=0.1)
        elif self.config.scheduler == 4:
            # For the aligned dataset
            self.config.num_epoch = 100 # for Registration Only 
            self.scheduler = MultiStepLR(
                optimizer, milestones=[150, 200], gamma=0.1)
        elif self.config.scheduler == 5:
            # For the aligned dataset
            self.config.num_epoch = 50# for 2D example only 
            self.scheduler = MultiStepLR(
                optimizer, milestones=[20, 40], gamma=0.1)

        self.optimizer = optimizer
        self.iter_idx = -1

        # build writer
        self._build_writer()


    def _build_writer(self):

        suffix = self.config.log_dir
        if suffix == "":
            suffix = "-".join(sys.argv)

        self.res_dir = os.path.join(self.config.res_dir, suffix)
        self.save_dir = self.config.save_dir
        if self.save_dir == "None":
            self.save_dir = self.res_dir
        else:
            self.res_dir = self.save_dir
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        
        tr_writer = SummaryWriter(log_dir=os.path.join(self.res_dir, self.config.mode))
        self.tr_writer = tr_writer

        self.checkpoint_file = os.path.join(self.res_dir, "checkpoint.pth")
        self.bestmodel_file = os.path.join(self.res_dir, "best_model.pth")
        self.log_fn = os.path.join(self.res_dir, "console.out")


    def _restore(self, pt_file):
        # Read checkpoint file.
        load_res = torch.load(pt_file)

        # Resume iterations
        self.iter_idx = load_res["iter_idx"]
        # Resume model
        self.model.load_state_dict(load_res["model"], strict=False)
        # Resume optimizer
        if self.config.mode == "train":
            self.optimizer.load_state_dict(load_res["optimizer"])


    def _restore_pretrain(self, pt_file):
        # Read checkpoint file.
        load_res = torch.load(pt_file)

        # Resume pretrained model
        self.model.load_state_dict(load_res["model"])


    def _save(self, pt_file):
        """ save models"""
        torch.save({
            "iter_idx": self.iter_idx,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict()},
            pt_file)


    def _check_nan_gradient(self):

        for param in self.model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).sum() > 0:
                    return True 
                    break
        return False 


    def train(self, data_loaders):
        self.model.train()

        data_loader_tr = data_loaders["train"]
        data_loader_va = data_loaders["valid"]

        # initialize parameters
        if not os.path.exists(self.checkpoint_file):
            print("training from scratch")
            cur_epoch = 0
            if self.config.pretrain_pt != "None":
                # Only load pretrained model. No optimizer
                dir_pretrain = "pretrained_model"
                pt_file = os.path.join(dir_pretrain, self.config.pretrain_pt, "best_model.pth")
                print(f"restore pretrained files {pt_file}")
                self._restore_pretrain(pt_file)
        else:
            print("restoring from {}".format(self.checkpoint_file))
            self._restore(self.checkpoint_file)

            # record cur_epoch
            fn = os.path.join(self.res_dir, "cur_epoch.txt")
            if os.path.exists(fn):
                with open(fn, "r") as f:
                    cur_epoch = int(f.read()) + 1
                for _ in np.arange(cur_epoch):
                    self.scheduler.step()
            else:
                cur_epoch = 0

        # Training loop
        best_va_acc = self.test(
            data_loader_va, mode="valid")["main_metric"]
        # fixed batch
        # data_fixed = next(iter(data_loader_tr))

        for epoch in range(cur_epoch, self.config.num_epoch):
            # update the learning rate after optimizer.step() as suggested by warning
            if hasattr(self, "scheduler"):
                self.scheduler.step()
            # # prefix for tqdm
            prefix = "Training: {:3d} / {:3d}".format(
                epoch, self.config.num_epoch)

            tic = time.time()
            losses = []
            # for data in data_loader_tr:
            for data in tqdm(data_loader_tr, prefix):
                self.iter_idx += 1

                # move tensor into cuda
                if self.config.use_cuda:
                    for key in data.keys():
                        data[key] = data[key].cuda()

                # inputs and labels
                in_dict = {} 
                in_dict["data"] = data
                in_dict["mode"] = "train"
                in_dict["iter_idx"] = self.iter_idx
                in_dict["writer"] = self.tr_writer 
                loss = self.model(in_dict) # return loss if mode="train"
                # Compute gradients
                loss.backward()
                # # Check Nan Gradient
                if self._check_nan_gradient():
                    # clear gradient everytime
                    self.optimizer.zero_grad()
                    print("Skip step of nan gradient")
                    continue
                # Update parameters
                self.optimizer.step()
                # Zero the parameter gradients in the optimizer
                self.optimizer.zero_grad()
                # print("loss: {}".format(loss))
                    
                # record and validation
                batch_size = len(data["pc"])
                losses += [loss.item() * batch_size]
                if self.iter_idx % self.config.rep_intv == 0:
                    self.tr_writer.add_scalar(
                        "loss", loss, global_step=self.iter_idx)
                    self._save(self.checkpoint_file)

            total_time_train_per_epoch = (time.time() - tic) / 3600.0
            loss_avg = np.array(losses).sum() / len(data_loader_tr.dataset)
            print_str = "Train Epoch %3d | loss %f | Time %.2fhr | lr %f" % \
                (epoch, loss_avg, total_time_train_per_epoch, self.optimizer.param_groups[0]['lr']) 
            print(print_str, flush=True)
            log(print_str, self.log_fn)
            
            # Valid after one epoch
            # Note that, on the datasets from AtlasNet, we use the model at the last step.   
            if (epoch + 1) % self.config.val_intv_epoch == 0:
                # mse: smaller means better
                tic = time.time()
                acc_indkt = self.test(
                    data_loader_va, mode="valid")["main_metric"]

                toal_time_test_per_epoch = (time.time() - tic) / 3600.0
                print_str = "Test Epoch %3d | loss %f | Time %.2fhr" % \
                    (epoch, acc_indkt, toal_time_test_per_epoch)
                print(print_str, flush=True)
                log(print_str, self.log_fn)

                # mse error
                if acc_indkt < best_va_acc:
                    self._save(self.bestmodel_file)
                    best_va_acc = acc_indkt 
                    # record best_va_acc
                    fn = os.path.join(self.res_dir, "best_va_acc.txt")
                    with open(fn, "w") as f:
                        f.write("{}\n".format(best_va_acc))
                        f.write("{}\n".format(epoch))

            # record cur_epoch
            fn = os.path.join(self.res_dir, "cur_epoch.txt")
            with open(fn, "w") as f:
                f.write("{}\n".format(epoch))


    def test(self, data_loader, mode="test"):
        """ test model"""
        self.model.eval()

        if mode == "test":
            if self.config.pt_file == "":
                # self._restore(self.bestmodel_file)
                # Note that, on the datasets from AtlasNet, we use the model at the last step.   
                self._restore(self.checkpoint_file)
            else:
                self._restore(self.config.pt_file)

        prefix = "testing"
        oas = []
        idx = 0
        vis_idx = [1, 2, 3]
        # for data in data_loader: 
        accs = {}
        feats = {}
        
        for data in tqdm(data_loader): 
            batch_size = len(data["pc"])
            idx += 1
            # move tensor into cuda
            if self.config.use_cuda:
                for key in data.keys():
                    data[key] = data[key].cuda()
            in_dict = {} 
            in_dict["data"] = data
            in_dict["mode"] = mode 
            in_dict["iter_idx"] = self.iter_idx
            in_dict["writer"] = self.tr_writer

            vis_fn = None
            if idx in vis_idx:
                vis_fn = f"{self.res_dir}/kp_att_{idx}_{self.iter_idx}.png"
            with torch.no_grad():
                # Apply the model to obtain scores (forward pass)
                accs_item, feats_item = self.model(
                    in_dict, vis_fn=vis_fn)
                for key in accs_item.keys():
                    if key not in accs:
                        accs[key] = []
                    accs[key] += [accs_item[key].item() * batch_size]

                for key in feats_item.keys():
                    if key not in feats:
                        feats[key] = []
                    feats[key] += [feats_item[key].cpu().numpy()]
    
        for key in accs.keys():
            accs[key] = np.array(accs[key]).sum() / len(data_loader.dataset)
            if mode == "valid":
                self.tr_writer.add_scalar(
                    "val_{}".format(key), accs[key], global_step=self.iter_idx)

        if mode == "test":
            if self.config.save_feat:
                # save feat only for multi cats
                h5fn_feat = os.path.join(self.save_dir, f"feat_{data_loader.dataset.mode}.h5")
                with h5py.File(h5fn_feat, "w") as f:
                    for key in feats.keys():
                        f[key] = np.array(feats[key])

            loss_txt = os.path.join(self.save_dir, f"loss_{data_loader.dataset.mode}.txt")
            with open(loss_txt, "w") as f:
                f.write("{}".format(accs["chamfer_error"]))

        accs["main_metric"] = accs["chamfer_error"]

        # Bring model to train mode 
        self.model.train()
        return accs


    def vis(self, data_loader, dump_vis_dir="dump"):
            """ test model"""
            self.model.eval()

            if self.config.pt_file == "":
                self._restore(self.checkpoint_file)
            else:
                self._restore(self.config.pt_file)

            prefix = "vis"
            dump_res = []
            idx = 0
            if self.config.vis_idx > 0:
                vis_idx = [self.config.vis_idx]
            else:
                vis_idx = [i for i in range(1, 4)]

            for data in tqdm(data_loader, desc=prefix): 
                idx += 1
                if idx not in vis_idx:
                    continue
                # move tensor into cuda
                if self.config.use_cuda:
                    for key in data.keys():
                        data[key] = data[key].cuda()
                in_dict = {} 
                in_dict["data"] = data
                in_dict["mode"] = "vis" 
                in_dict["iter_idx"] = self.iter_idx
                in_dict["writer"] = self.tr_writer 

                vis_dir = f"{self.save_dir}/dump_vis_{idx}"
                if not os.path.exists(vis_dir):
                    os.makedirs(vis_dir)

                with torch.no_grad():
                    self.model.vis_single(in_dict, vis_dir)
#
# network.py ends here