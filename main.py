#!/usr/bin/env python3
import numpy as np
from network import Network
from torch.utils.data import DataLoader
from config import get_config, print_usage
from data_utils.ShapeNetLoader import ShapeNetLoader

def main(config):
    """The main function."""
    network = Network(config)
    if config.dataset == "modelnet40":
        dataset = ModelNet40DataLoader
    elif config.dataset == "shapenet":
        dataset = ShapeNetLoader
        if config.input_feat in ["ppf"] \
            or config.pose_code in ["weighted_qt"]:
            require_normal = True
        else:
            require_normal = False

    if config.mode == "train":
            
        data_loaders = {}
        # load training data
        data_tr = dataset(
            data_dump_folder=config.data_dump_folder, indim=config.indim, id=config.cat_id, require_normal=require_normal,
            num_pts=config.num_pts, mode="train", jitter_type=config.pc_jitter_type)

        data_loader_tr = DataLoader(
            dataset=data_tr,
            batch_size=config.batch_size,
            num_workers=config.worker,
            shuffle=True,
            pin_memory=True,
        )
        data_loaders["train"] = data_loader_tr

        # load valid data 
        data_va = dataset(
            data_dump_folder=config.data_dump_folder, indim=config.indim, id=config.cat_id, require_normal=require_normal,
            num_pts=config.num_pts, mode="test", jitter_type=config.pc_jitter_type)

        data_loader_va = DataLoader(
            dataset=data_va,
            batch_size=config.test_batch_size,
            num_workers=config.worker,
            shuffle=False,
            pin_memory=True,
        )
        data_loaders["valid"] = data_loader_va
        network.train(data_loaders)

    elif config.mode == "test":
        # load valid data
        data_te = dataset(
            data_dump_folder=config.data_dump_folder, indim=config.indim, freeze_data=True, id=config.cat_id, require_normal=require_normal,
            num_pts=config.num_pts, mode="test", jitter_type=config.pc_jitter_type)
        data_loader_te = DataLoader(
            dataset=data_te,
            batch_size=config.test_batch_size,
            num_workers=config.worker,
            shuffle=False,
            pin_memory=True,
        )
        accs = network.test(data_loader_te)
        print(f"test: {accs}")
    elif config.mode == "vis":
        # load valid data
        data_te = dataset(
            data_dump_folder=config.data_dump_folder, indim=config.indim, freeze_data=True, id=config.cat_id, require_normal=require_normal,
            num_pts=config.num_pts, mode="test", jitter_type=config.pc_jitter_type)
        data_loader_te = DataLoader(
            dataset=data_te,
            batch_size=config.test_batch_size,
            num_workers=0,
            shuffle=False,
            pin_memory=False,
        )
        accs = network.vis(data_loader_te)
        print(f"test: {accs}")
    else:
        raise ValueError("Unknown run mode \"{}\"".format(config.mode))


if __name__ == "__main__":

    # ----------------------------------------
    # Parse configuration
    config, unparsed = get_config()
    # If we have unparsed arguments, print usage and exit
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)
#
# main.py ends here
