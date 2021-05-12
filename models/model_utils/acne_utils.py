import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from config import get_config, print_usage
config, unparsed = get_config()

class KoutLayer(nn.Module):
    def __init__(self, inc, num_g, shift=False, outc=None, bn_type="gn"):
        super(KoutLayer, self).__init__()
        print("num K: {}".format(num_g))

        self.conv_att = nn.Conv2d(inc, num_g, 1)        

        self.num_g = num_g
        # self.w_linear = nn.Parameter(torch.randn(1, num_g, outc, inc))
        if outc is not None:
            self.linear = nn.Conv2d(inc, outc, 1)
            if bn_type == "gn":
                self.norm = nn.GroupNorm(32, outc)
            elif bn_type == "bn":
                self.norm = nn.BatchNorm2d(outc)
            else:
                self.norm = nn.Identity()
        else:
            self.linear = None

    def forward(self, x, return_att=False):
        
        attention = self.conv_att(x)
        a = torch.softmax(attention, dim=1) # BKN1
        # ACN
        a = a[:, None]# B1GN1

        if config.a_norm_eps == "clamp":
            pai = a.sum(dim=3, keepdim=True) # B1K11
            a_norm = a / torch.clamp(pai, min=1e-3)
        elif config.a_norm_eps == "none":
            a_norm = a / a.sum(dim=3, keepdim=True)
        elif config.a_norm_eps == "eps":
            a_norm = a / (a.sum(dim=3, keepdim=True) + 1e-8)
        else:
            raise NotImplementedError
        x = x[:, :, None] # BC1N1
        mean = torch.sum(x * a_norm, dim=3) # B*C*num_group*1
        if self.linear is not None:
            mean = F.relu(self.norm(self.linear(mean)))

        if return_att:
            return mean, a
        else:
            return mean 



class ACN2dMultiBranch(nn.Module):
    def __init__(self, inc, num_g=1, atten_opt="softmax", rescale=None, eps=1e-3, reg=False, bin_score=None):
        super(ACN2dMultiBranch, self).__init__()
        print("num_head: {}".format(num_g))
        self.atten_opt = atten_opt
        self.num_g = num_g
        self.eps = eps
        if self.atten_opt == "sigmoid_softmax":
            self.conv = nn.Conv2d(inc, 2*num_g, 1)
        elif self.atten_opt == "softmax":
            self.conv = nn.Conv2d(inc, 1*num_g, 1)

        self.rescale = rescale
        if self.rescale is not None:
            self.weight_bias = nn.Parameter(torch.ones(1, 1, inc, num_g))
            self.bias_bias = nn.Parameter(torch.zeros(1, 1, inc, num_g))
        self.reg = reg 
        if self.reg:
            self.scale = torch.nn.Parameter(torch.FloatTensor(1, 1, num_g, 1))
            self.scale.data = torch.ones(1) * 0.1
            self.register_parameter("reg_scale", self.scale)
        
    def forward(self, x, context_vec=None):

        if context_vec != None:
            # context_vec with BC11
            a = torch.matmul(
                x.permute(0, 2, 3, 1), context_vec.transpose(2, 1))
            a = torch.softmax(a.view(a.shape[0], -1), dim=1)
        else:
            if self.atten_opt == "sigmoid_softmax":
                a_l = torch.sigmoid(attention[:, :self.num_g, :, :])
                a_g = torch.softmax(attention[:, self.num_g:, :, :], dim=2)
                # Merge a_l and a_g
                a = a_l * a_g
                a = a / (a.sum(dim=-1, keepdim=True) + self.eps)
            elif self.atten_opt == "softmax":
                attention = self.conv(x)
                if self.training and self.reg:
                    idx = np.random.choice(
                        range(x.shape[2]), self.num_g, replace=False)
                    x_select = x[:, :, idx] # BCM1 
                    coef = torch.matmul(
                        x_select.squeeze(3).transpose(2, 1), x.squeeze(3)) # BMN
                    attention += self.scale * coef[..., None]
                a = torch.softmax(attention, dim=1)
                # no need to do the normalization
            elif self.atten_opt == "None":
                a = torch.ones(x.shape[0], self.num_g, 
                               x.shape[2], 1, dtype=torch.float32,
                               device=x.device)
                a = a / (a.sum(dim=-1, keepdim=True) + self.eps)
            else:
                raise ValueError("wrong atten_opt")

        # ACN
        a = a[:, None] # B1GN1
        
        if config.a_norm_eps == "clamp":
            pai = a.sum(dim=3, keepdim=True) # B1K11
            a_norm = a / torch.clamp(pai, min=1e-3)
        elif config.a_norm_eps == "none":
            a_norm = a / a.sum(dim=3, keepdim=True)
        elif config.a_norm_eps == "eps":
            a_norm = a / (a.sum(dim=3, keepdim=True) + 1e-8)
        else:
            raise NotImplementedError
            
        # a_j = a.mean(dim=3, keepdim=True) # B1G11
        x = x[:, :, None] # BC1N1
        mean = torch.sum(x * a_norm, dim=3, keepdim=True) # B*C*num_group*1*1
        out = x - mean
        std = torch.sqrt(
            torch.sum(a_norm * out**2, dim=3, keepdim=True) + self.eps)
        out = out / std
        # out = out.squeeze(2)
        out = torch.sum(out * a, dim=2) # BCN1

        # rescale
        # 11CG x B1GN
        if self.rescale is not None:
            weight_bias = torch.matmul(self.weight_bias, a.squeeze(-1)).permute(0, 2, 3,1) # B1CN 
            bias_bias = torch.matmul(self.bias_bias, a.squeeze(-1)).permute(0, 2, 3, 1) # B1CN
            out = out * weight_bias + bias_bias
        return out


class ConvLayer(nn.Module):
    def __init__(self, inc, outc, cn_type="acn_g", cn_pos="post", bn="gn", reg=False, act="relu", bin_score=None):
        super(ConvLayer, self).__init__()
        self.layer = nn.Sequential()

        assert cn_pos in ["post", "pre"]
        if cn_pos == "post":
            self.layer.add_module(
                "conv", nn.Conv2d(inc, outc, 1, 1))
            inc = outc
        
        print("cn_type: {}".format(cn_type))

        if cn_type.startswith("acn_b"):
            num_g = int(cn_type.split("-")[-1])
            self.layer.add_module(
                "cn", ACN2dMultiBranch(inc, num_g=num_g, reg=reg, bin_score=bin_score))
        elif cn_type.startswith("acn_vanilla_b"):
            num_g = int(cn_type.split("-")[-1])
            self.layer.add_module(
                "cn", ACN2dMultiBranchVanilla(inc, num_g))
        elif cn_type.startswith("acn_g"):
            num_g = int(cn_type.split("-")[-1])
            self.layer.add_module(
                "cn", ACN2dGroup(inc, num_g))
        elif cn_type.startswith("None"):
            pass
        else: 
            raise NotImplementedError

        if cn_pos == "pre":
            inc = outc
            self.layer.add_module(
                "conv", nn.Conv2d(inc, outc, 1, 1))

        if bn == "gn":
            self.layer.add_module(
                "bn", nn.GroupNorm(32, outc))
        elif bn == "bn":
            self.layer.add_module(
                "bn", nn.BatchNorm2d(outc)
            )
        elif bn == "None":
            pass
        else:
            raise NotImplementedError
        print("bn type: {}".format(bn))
        if act == "relu":
            self.layer.add_module(
                "relu", nn.ReLU(inplace=True))
        elif act == "sine":
            self.layer.add_module("sine", Sine())
    
    def forward(self, x):
        return self.layer(x)


class ResBlock(nn.Module):

    def __init__(self, inc, outc, num_inner=2, cn_type="acn_b-16", bn="gn", reg=False, act="relu", bin_score=None):
        super(ResBlock, self).__init__()
        if inc != outc:
            self.pass_through = ConvLayer(
                inc, outc, cn_type=cn_type, bn=bn, act=act, bin_score=bin_score)
        else:
            self.pass_through = None

        self.conv_layers = nn.Sequential()
        for _i in range(num_inner):
            self.conv_layers.add_module(
                "conv-{}".format(_i), ConvLayer(outc, outc, cn_type=cn_type, bn=bn, reg=reg, act=act, bin_score=bin_score)
            )

    def forward(self, x):
        if self.pass_through is not None:
            x = self.pass_through(x)
        return x + self.conv_layers(x)



class AcneKpEncoder(nn.Module):
    def __init__(self, config, indim, out_dim=None):
        """Acne for embedding
        """
        super(AcneKpEncoder, self).__init__()
        self.config = config
        if indim is not None:
            inc = indim
        
        net_depth = self.config.acne_net_depth
        num_g = self.config.acne_num_g

        outc = self.config.acne_dim

        bn_type = self.config.acne_bn_type
        cn_type = "{}-{}".format(self.config.cn_type, num_g)

        self.layers = nn.Sequential()
        for i in range(net_depth):
            self.layers.add_module(
                "layer-{}".format(i), ResBlock(inc, outc, num_inner=2, cn_type=cn_type, bn=bn_type))
            inc = outc

        self.layers.add_module(
            "vote-{}".format(i),
            KoutLayer(
                inc, num_g, outc=out_dim, bn_type=bn_type)
            )


    def forward(self, xyz, att_aligner=None, return_att=False):

        B = xyz.shape[0]
        x = self.layers[:-1](xyz)# BCK1
        if att_aligner is not None:
            pai = att_aligner.sum(dim=3, keepdim=True) # B1K11
            a_norm = att_aligner / torch.clamp(pai, min=1e-3)
            x = x[:, :, None] # BC1N1
            mean = torch.sum(x * a_norm, dim=3) # B*C*num_group*1
            ret = (mean, att_aligner)
        else:
            ret = self.layers[-1](x, return_att=return_att) 
        
        return ret
