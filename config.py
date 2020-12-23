import argparse

# ----------------------------------------
# Global variables within this script
arg_lists = []
parser = argparse.ArgumentParser()


# ----------------------------------------
# Some nice macros to be used for arparse
def str2bool(v):
    return v.lower() in ("true", "1")

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

# ----------------------------------------
# Arguments for the main program
main_arg = add_argument_group("Main")

main_arg.add_argument("--mode", type=str,
                      default="train",
                      choices=["train", "test", "valid", "vis", "eval_regist"],
                      help="Run mode")

main_arg.add_argument("--data_dump_folder", type=str,
                      default="data_dump",
                      help="data_dump_folder saving the data")

main_arg.add_argument("--pc_jitter_type", type=str,
                      default="None",
                      help="pc jittter type")

main_arg.add_argument("--dataset", type=str,
                      default="shapenet",
                      help="dataset")

main_arg.add_argument("--cat_id", type=int,
                      default=9,
                      help="data category")

main_arg.add_argument("--suffix", type=str,
                      default="",
                      help="For ease of naming logging folders")
main_arg.add_argument("--cn_type", type=str,
                      default="acn_b",
                      help="Encoder context normalization type")
main_arg.add_argument("--pt_file", type=str,
                      default="",
                      help="pt file")
main_arg.add_argument("--feat_net", type=str,
                      default="AcneKpEncoder",
                      help="Encoder")

# ----------------------------------------
# Arguments for model 
model_arg = add_argument_group("model")
model_arg.add_argument("--ae_decoder", type=str,
                       default="KpDecoder",
                       help="Decoder to reconstruct point clouds")
model_arg.add_argument("--input_feat", type=str,
                       default="None",
                       help="type of input feature")
model_arg.add_argument("--pose_code", type=str,
                       default="nl-noR_T",
                       choices=[None, "weighted_qt", "l-localRT", "l-RT", "nl-UStV", "nl-U", "nl-T", "nl-noR_T", "nl-LRF_T", "nl-lq_T"],
                       help="pose type of capsule")
model_arg.add_argument("--indim", type=int,
                       default=3,
                       help="input dimension")
model_arg.add_argument("--num_pts", type=int,
                       default=1024,
                       help="num of pts")
model_arg.add_argument("--emb_dims", type=int,
                       default=512,
                       help="emb dims")

model_arg.add_argument("--decoder_bottleneck_size", type=int,
                       default=1280,
                       help="decoder dims")

model_arg.add_argument("--acne_dim", type=int,
                       default=128,
                       help="emb dims for acne")
model_arg.add_argument("--acne_num_g", type=int,
                       default=10,
                       help="num_g")
model_arg.add_argument("--acne_net_depth", type=int,
                       default=3,
                       help="acne_net_depth")
model_arg.add_argument("--acne_out_dim", type=int,
                       default=0,
                       help="acne_out_dim")
model_arg.add_argument("--acne_bn_type", type=str,
                       default="bn",
                       help="bn type")
model_arg.add_argument("--bin_score", type=str,
                       default=None,
                       help="bin score")
model_arg.add_argument("--mean_type", type=str,
                       default="q",
                       help="how to normalize")
model_arg.add_argument("--acn_mean_type", type=str,
                       default="q",
                       help="how to normalize")
model_arg.add_argument("--acne_backbone", type=str,
                       default="None",
                       help="ablation study with backbone network")
model_arg.add_argument("--acn_reg", type=str2bool,
                       default=False,
                       help="whether add regularizer into normalizer")
model_arg.add_argument("--use_pose", type=str2bool,
                       default=True,
                       help="use pose")
model_arg.add_argument("--add_noise_kps", type=str2bool,
                       default=False,
                       help="noise to kp")
# model_arg.add_argument("--loss_", type=float,
#                        default=0.0,
#                        help="add rotation equivariance")
model_arg.add_argument("--loss_align_kp_consistency", type=float,
                       default=0.0,
                       help="add rotation equivariance")
model_arg.add_argument("--loss_kps_ref_consistency", type=float,
                       default=0.0,
                       help="Ref keypoints should be same.")
model_arg.add_argument("--loss_beta", type=float,
                       default=0.0,
                       help="add rotation equivariance")
model_arg.add_argument("--loss_reg_att_f", type=float,
                       default=0.0,
                       help="regularizers for the final attention")
model_arg.add_argument("--loss_aligner", type=float,
                       default=1.0,
                       help="regularizers for the final attention")
model_arg.add_argument("--loss_equi_assign", type=float,
                       default=0.0,
                       help="regularizers for the final attention")

model_arg.add_argument("--loss_equi_r", type=float,
                       default=0.0,
                       help="add rotation equivariance")
model_arg.add_argument("--loss_procruste", type=float,
                       default=0.0,
                       help="add rotation equivariance")
model_arg.add_argument("--loss_separation", type=float,
                       default=0.0,
                       help="add rotation equivariance")
model_arg.add_argument("--loss_2cps_mu", type=float,
                       default=0.0,
                       help="add rotation equivariance")
model_arg.add_argument("--loss_ref_kp_can", type=float,
                       default=0.0,
                       help="add rotation equivariance")
model_arg.add_argument("--separation_margin", type=float,
                       default=0.2,
                       help="separation margin")
model_arg.add_argument("--procruste_noise", type=float,
                       default=1e-8,
                       help="separation margin")
model_arg.add_argument("--random_range", type=str,
                       default="uni-180-0.2",
                       help="random range")
model_arg.add_argument("--pretrain_pt", type=str,
                       default="None",
                       help="norm type for decoder")
model_arg.add_argument("--out_pose_grad", type=str,
                       default="pc_can",
                       help="where to multiply pose")
model_arg.add_argument("--decoder_grid", type=str,
                       default="learnable",
                       help="type of decoder grid")
model_arg.add_argument("--acne_input_layer", type=str,
                       default="None",
                       choices=["conv", "conv_cn_bn_relu", "conv_bn_relu"],
                       help="type of input layers")
model_arg.add_argument("--acne_num_inner", type=int,
                       default=2,
                       help="emb dims for acne")
model_arg.add_argument("--KpDecoderPose", type=str2bool,
                       default=False,
                       help="Whether bring points to local")
model_arg.add_argument("--aligner", type=str,
                       default="init",
                       help="Whether bring points to local")
model_arg.add_argument("--shift", type=str2bool,
                       default=False,
                       help="voting with euclidean")
model_arg.add_argument("--conf_corr", type=str2bool,
                       default=False,
                       help="Whether bring points to local")
model_arg.add_argument("--trans_mid", type=str,
                       default="None",
                       help="transform the input")
model_arg.add_argument("--pose_block", type=str,
                       default="None",
                       help="transform the input")
model_arg.add_argument("--spatial_var_norm", type=str,
                       default="l1",
                       help="transform the input")
model_arg.add_argument("--KpDecoderMlp", type=str,
                       default="nonshare",
                       help="Whether bring points to local")
model_arg.add_argument("--num_classes", type=int,
                       default=40,
                       help="num classes")
model_arg.add_argument("--num_view", type=int,
                       default=2,
                       help="num classes")
model_arg.add_argument("--vis_id", type=int,
                       default=-1,
                       help="num classes")
model_arg.add_argument("--vis_idx", type=int,
                       default=-1,
                       help="num classes")
model_arg.add_argument("--act", type=str,
                       default="relu",
                       help="activation")
model_arg.add_argument("--att_type_out", type=str,
                       default="gmm",
                       help="attention type of the kout layer")
model_arg.add_argument("--ref_pcd_fn", type=str,
                       default="None",
                       help="filename of ref_pcd_fn")
model_arg.add_argument("--patch_pos", type=str,
                       default="center_att",
                       help="filename of ref_pcd_fn")
model_arg.add_argument("--ref_kp_type", type=str,
                       default="None",
                       help="the way of generating reference kp")
model_arg.add_argument("--num_ref_pcd", type=int,
                       default=1,
                       help="num ")
model_arg.add_argument("--num_ref_kp", type=int,
                       default=0,
                       help="num ")
model_arg.add_argument("--a_norm_eps", type=str,
                       default="clamp",
                       help="how to use compute the a_norm")

# ----------------------------------------
# Arguments for training
train_arg = add_argument_group("Training")

train_arg.add_argument("--model", type=str,
                      default="AcneAe",
                      help="model name")

train_arg.add_argument("--scheduler", type=int,
                       default=1,
                       help="Adjust learning rate with MultiStepLR")

train_arg.add_argument("--noise_ratio", type=float,
                       default=0.2,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--learning_rate", type=float,
                       default=1e-3,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--weight_decay", type=float,
                       default=0,
                       help="Learning rate (gradient step size)")

train_arg.add_argument("--batch_size", type=int,
                       default=16,
                       help="Size of each training batch")

train_arg.add_argument("--test_batch_size", type=int,
                       default=1,
                       help="Size of each training batch")

train_arg.add_argument("--num_points", type=int,
                       default=1024,
                       help="The number of points for point clouds")

train_arg.add_argument("--num_epoch", type=int,
                       default=450,
                       help="Number of epochs to train")

train_arg.add_argument("--val_intv", type=int,
                       default=5000,
                       help="Validation interval")

train_arg.add_argument("--val_intv_epoch", type=int,
                       default=10,
                       help="Validation interval")

train_arg.add_argument("--rep_intv", type=int,
                       default=100,
                       help="Report interval")

train_arg.add_argument("--log_dir", type=str,
                       default="",
                       help="Directory to save logs")

train_arg.add_argument("--res_dir", type=str,
                       default="./logs",
                       help="Directory to save current model")
train_arg.add_argument("--save_dir", type=str,
                       default="None",
                       help="Directory to save current model")

train_arg.add_argument("--swap_code", type=str,
                       default="None",
                       help="swap code before decoder")

train_arg.add_argument("--pc_align", type=str,
                       default="x",
                       help="supervision")

train_arg.add_argument("--resume", type=str2bool,
                       default=False,
                       help="Whether to resume training "
                       "from existing checkpoint")

train_arg.add_argument("--save_feat", type=str2bool,
                       default=False,
                       help="Whether to resume training "
                       "from existing checkpoint")

train_arg.add_argument("--att_chamfer", type=str2bool,
                       default=False,
                       help="Whether to resume training "
                       "from existing checkpoint")
train_arg.add_argument("--rt_grid", type=str2bool,
                       default=False,
                       help="Whether to resume training "
                       "from existing checkpoint")

train_arg.add_argument("--worker", type=int,
                       default=20,
                       help="number of workers")

train_arg.add_argument("--grid_dim", type=int,
                       default=10,
                       help="number of workers")

train_arg.add_argument("--loss_entropy", type=float,
                       default=0,
                       help="loss for entropy")

train_arg.add_argument("--loss_trans_mid", type=float,
                       default=0,
                       help="loss for entropy")

train_arg.add_argument("--loss_chamfer_cp", type=float,
                       default=0,
                       help="loss for entropy")

train_arg.add_argument("--loss_2cps", type=float,
                       default=5,
                       help="loss for entropy")

train_arg.add_argument("--loss_vol_cp", type=float,
                       default=0,
                       help="loss for entropy")

train_arg.add_argument("--loss_cov_pts", type=float,
                       default=0,
                       help="loss for entropy")

train_arg.add_argument("--loss_att_amount", type=float,
                       default=1e-3,
                       help="loss for entropy")

train_arg.add_argument("--loss_orth_att", type=float,
                       default=0,
                       help="loss for entropy")

train_arg.add_argument("--loss_chamfer_cp_side", type=float,
                       default=0,
                       help="Constraint that center point should be around original points")

train_arg.add_argument("--loss_trans_mid_can", type=float,
                       default=0,
                       help="loss for entropy")

train_arg.add_argument("--loss_volume", type=float,
                       default=1,
                       help="loss for entropy")

train_arg.add_argument("--loss_reconstruction", type=float,
                       default=1,
                       help="loss forreconstruction")

train_arg.add_argument("--loss_decode_cp", type=float,
                       default=0,
                       help="loss forreconstruction")

train_arg.add_argument("--use_cuda", type=str2bool,
                       default=True,
                       help="cuda seems slower?")

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed


def print_usage():
    parser.print_usage()

#
# config.py ends here