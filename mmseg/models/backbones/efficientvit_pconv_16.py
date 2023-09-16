# --------------------------------------------------------
# EfficientViT Model Architecture for Downstream Tasks
# Copyright (c) 2022 Microsoft
# Written by: Xinyu Liu
# --------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import itertools

from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite, DropPath, to_2tuple

import numpy as np
import itertools
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from torch import Tensor

# from mmcv_custom import load_checkpoint, _load_checkpoint, load_state_dict
# from mmdet.utils import get_root_logger
# from mmdet.models.builder import BACKBONES
from torch.nn.modules.batchnorm import _BatchNorm


class Conv2d_BN(torch.nn.Sequential):
    def __init__(self, a, b, ks=1, stride=1, pad=0, dilation=1,
                 groups=1, bn_weight_init=1, resolution=-10000):
        super().__init__()
        self.add_module('c', torch.nn.Conv2d(
            a, b, ks, stride, pad, dilation, groups, bias=False))
        self.add_module('bn', torch.nn.BatchNorm2d(b))
        torch.nn.init.constant_(self.bn.weight, bn_weight_init)
        torch.nn.init.constant_(self.bn.bias, 0)

    @torch.no_grad()
    def fuse(self):
        c, bn = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps)**0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation, groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m
    
class Partial_conv3(nn.Module):

    def __init__(self, dim, n_div, forward):
        super().__init__()
        self.dim_conv3 = dim // n_div
        self.dim_untouched = dim - self.dim_conv3
        self.partial_conv3 = nn.Conv2d(self.dim_conv3, self.dim_conv3, 3, 1, 1, bias=False)

        if forward == 'slicing':
            self.forward = self.forward_slicing
        elif forward == 'split_cat':
            self.forward = self.forward_split_cat
        else:
            raise NotImplementedError

    def forward_slicing(self, x: Tensor) -> Tensor:
        # only for inference
        x = x.clone()   # !!! Keep the original input intact for the residual connection later
        x[:, :self.dim_conv3, :, :] = self.partial_conv3(x[:, :self.dim_conv3, :, :])

        return x

    def forward_split_cat(self, x: Tensor) -> Tensor:
        # for training/inference
        # print(x.shape)

        x1, x2 = torch.split(x, [self.dim_conv3, self.dim_untouched], dim=1)
        
        x1 = self.partial_conv3(x1)
        
        # x = torch.cat((x1, x2), 1)

        return x1, x2




class BN_Linear(torch.nn.Sequential):
    def __init__(self, a, b, bias=True, std=0.02):
        super().__init__()
        self.add_module('bn', torch.nn.BatchNorm1d(a))
        self.add_module('l', torch.nn.Linear(a, b, bias=bias))
        trunc_normal_(self.l.weight, std=std)
        if bias:
            torch.nn.init.constant_(self.l.bias, 0)

    @torch.no_grad()
    def fuse(self):
        bn, l = self._modules.values()
        w = bn.weight / (bn.running_var + bn.eps)**0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps)**0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m

def replace_batchnorm(net):
    for child_name, child in net.named_children():
        if hasattr(child, 'fuse'):
            setattr(net, child_name, child.fuse())
        elif isinstance(child, torch.nn.BatchNorm2d):
            setattr(net, child_name, torch.nn.Identity())
        else:
            replace_batchnorm(child)
            

class PatchMerging(torch.nn.Module):
    def __init__(self, dim, out_dim, input_resolution):
        super().__init__()
        hid_dim = int(dim * 4)
        self.conv1 = Conv2d_BN(dim, hid_dim, 1, 1, 0, resolution=input_resolution)
        self.act = torch.nn.ReLU()
        self.conv2 = Conv2d_BN(hid_dim, hid_dim, 3, 2, 1, groups=hid_dim, resolution=input_resolution)
        self.se = SqueezeExcite(hid_dim, .25)
        self.conv3 = Conv2d_BN(hid_dim, out_dim, 1, 1, 0, resolution=input_resolution // 2)

    def forward(self, x):
        x = self.conv3(self.se(self.act(self.conv2(self.act(self.conv1(x))))))
        return x


class Residual(torch.nn.Module):
    def __init__(self, m, drop=0.):
        super().__init__()
        self.m = m
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.m(x) * torch.rand(x.size(0), 1, 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.m(x)


class FFN(torch.nn.Module):
    def __init__(self, ed, h, resolution):
        super().__init__()
        self.pw1 = Conv2d_BN(ed, h, resolution=resolution)
        self.act = torch.nn.ReLU()
        self.pw2 = Conv2d_BN(h, ed, bn_weight_init=0, resolution=resolution)

    def forward(self, x):
        x = self.pw2(self.act(self.pw1(x)))
        return x


class CascadedGroupAttention(torch.nn.Module):
    r""" Cascaded Group Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution, correspond to the window size.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 reduce_ratio=16,
                 resolution=14,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        # self.key_dim = key_dim
        # self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        self.dim = dim
        self.reduce_dim = int (dim // reduce_ratio)
        self.qkv_dim = 3 * self.reduce_dim
        self.reduce_ratio = reduce_ratio

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(self.reduce_dim, self.qkv_dim, resolution=resolution))
            dws.append(Conv2d_BN(self.reduce_dim, self.reduce_dim, kernels[i], 1, kernels[i]//2, groups=self.reduce_dim, resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        pconvs = []
        for i in range(num_heads):
            pconvs.append(Partial_conv3(dim=dim, n_div=reduce_ratio, forward='split_cat'))
        self.pconvs = torch.nn.ModuleList(pconvs)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.dim * num_heads, self.dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        N = len(points)
        attention_offsets = {}
        idxs = []
        for p1 in points:
            for p2 in points:
                offset = (abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))
                if offset not in attention_offsets:
                    attention_offsets[offset] = len(attention_offsets)
                idxs.append(attention_offsets[offset])
        self.attention_biases = torch.nn.Parameter(
            torch.zeros(num_heads, len(attention_offsets)))
        self.register_buffer('attention_bias_idxs',
                             torch.LongTensor(idxs).view(N, N))

    @torch.no_grad()
    def train(self, mode=True):
        super().train(mode)
        if mode and hasattr(self, 'ab'):
            del self.ab
        else:
            self.ab = self.attention_biases[:, self.attention_bias_idxs]

    def forward(self, x):  # x (B,C,H,W)
        B, C, H, W = x.shape
        # print(self.dim)
        # print(self.attn_ratio)
        
       
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        # feats_in = x.chunk(len(self.qkvs), dim=1)
        feats_out = []
        feat = x
        for i, qkv in enumerate(self.qkvs):
            feat, x2= self.pconvs[i](feat)
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.reduce_dim, self.reduce_dim, self.reduce_dim], dim=1) # B, C/h, H, W
            # q, k, v = feat.view(B, -1, H, W)
            # print('q', q.shape)
            # print('k', k.shape)
            # print('v', v.shape)
            q = self.dws[i](q)
            # print('q', q.shape)
            # print('k', k.shape)
            # print('v', v.shape)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
            
            attn = (
                (q.transpose(-2, -1) @ k) * self.scale
                +
                (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1) # BNN
            # print(attn.transpose(-2, -1).shape)
            # print(v.shape)
            # print((v @ attn.transpose(-2, -1)).shape)
            feat = (v @ attn.transpose(-2, -1)).view(B, self.reduce_dim, H, W) # BCHW
            # print(feat.shape)
            feat = torch.cat((feat, x2), dim=1)
            # print(feat.shape)
            feats_out.append(feat)
        x = self.proj(torch.cat(feats_out, 1))
        return x


class LocalWindowAttention(torch.nn.Module):
    r""" Local Window Attention.

    Args:
        dim (int): Number of input channels.
        key_dim (int): The dimension for query and key.
        num_heads (int): Number of attention heads.
        attn_ratio (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, dim, key_dim, num_heads=8,
                 attn_ratio=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution
        
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                attn_ratio=attn_ratio, 
                                resolution=window_resolution,
                                kernels=kernels,)

    def forward(self, x):
    
        B, C, H, W = x.shape
        
               
        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            pad_b = (self.window_resolution - H %
                     self.window_resolution) % self.window_resolution
            pad_r = (self.window_resolution - W %
                     self.window_resolution) % self.window_resolution
            padding = pad_b > 0 or pad_r > 0

            if padding:
                x = torch.nn.functional.pad(x, (0, 0, 0, pad_r, 0, pad_b))

            pH, pW = H + pad_b, W + pad_r
            nH = pH // self.window_resolution
            nW = pW // self.window_resolution
            # window partition, BHWC -> B(nHh)(nWw)C -> BnHnWhwC -> (BnHnW)hwC -> (BnHnW)Chw
            x = x.view(B, nH, self.window_resolution, nW, self.window_resolution, C).transpose(2, 3).reshape(
                B * nH * nW, self.window_resolution, self.window_resolution, C
            ).permute(0, 3, 1, 2)
            x = self.attn(x)
            # window reverse, (BnHnW)Chw -> (BnHnW)hwC -> BnHnWhwC -> B(nHh)(nWw)C -> BHWC
            x = x.permute(0, 2, 3, 1).view(B, nH, nW, self.window_resolution, self.window_resolution,
                       C).transpose(2, 3).reshape(B, pH, pW, C)

            if padding:
                x = x[:, :H, :W].contiguous()

            x = x.permute(0, 3, 1, 2)

        return x


class EfficientViTBlock(torch.nn.Module):
    """ A basic EfficientViT building block.

    Args:
        type (str): Type for token mixer. Default: 's' for self-attention.
        ed (int): Number of input channels.
        kd (int): Dimension for query and key in the token mixer.
        nh (int): Number of attention heads.
        ar (int): Multiplier for the query dim for value dimension.
        resolution (int): Input resolution.
        window_resolution (int): Local window resolution.
        kernels (List[int]): The kernel size of the dw conv on query.
    """
    def __init__(self, type,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5],):
        super().__init__()
            
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar, \
                    resolution=resolution, window_resolution=window_resolution, kernels=kernels))
                
        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))
    

@MODELS.register_module()
class EfficientPcViT16(BaseModule):
    def __init__(self, img_size=400,
                 patch_size=16,
                 frozen_stages=0,
                 in_chans=3,
                 stages=['s', 's', 's'],
                 embed_dim=[64, 128, 192],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 4, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 pretrained=None,
                 distillation=False,):
        super().__init__()
        resolution = img_size
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1, resolution=resolution // 2), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1, resolution=resolution // 4), torch.nn.ReLU(),
                           Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1, resolution=resolution // 8))

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i+1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                #('Subsample' stride)
                blk = eval('self.blocks' + str(i+2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)),))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1], resolution=resolution)),
                                    Residual(FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)),))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)
        
        self.frozen_stages = frozen_stages # freeze the patch embedding
        self._freeze_stages()

        if pretrained is not None:
            self.init_weights(pretrained=pretrained)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

    # def init_weights(self, pretrained=None):
    #     """Initialize the weights in backbone.

    #     Args:
    #         pretrained (str, optional): Path to pre-trained weights.
    #             Defaults to None.
    #     """

    #     if isinstance(pretrained, str):
    #         logger = get_root_logger()
    #         checkpoint = _load_checkpoint(pretrained, map_location='cpu')
            
    #         if not isinstance(checkpoint, dict):
    #             raise RuntimeError(
    #                 f'No state_dict found in checkpoint file {filename}')
    #         # get state_dict from checkpoint
    #         if 'state_dict' in checkpoint:
    #             state_dict = checkpoint['state_dict']
    #         elif 'model' in checkpoint:
    #             state_dict = checkpoint['model']
    #         else:
    #             state_dict = checkpoint
    #         # strip prefix of state_dict
    #         if list(state_dict.keys())[0].startswith('module.'):
    #             state_dict = {k[7:]: v for k, v in state_dict.items()}
            
    #         model_state_dict = self.state_dict()
    #         # bicubic interpolate attention_biases if not match

    #         rpe_idx_keys = [
    #             k for k in state_dict.keys() if "attention_bias_idxs" in k]
    #         for k in rpe_idx_keys:
    #             print("deleting key: ", k)
    #             del state_dict[k]

    #         relative_position_bias_table_keys = [
    #             k for k in state_dict.keys() if "attention_biases" in k]
    #         for k in relative_position_bias_table_keys:
    #             relative_position_bias_table_pretrained = state_dict[k]
    #             relative_position_bias_table_current = model_state_dict[k]
    #             nH1, L1 = relative_position_bias_table_pretrained.size()
    #             nH2, L2 = relative_position_bias_table_current.size()
    #             if nH1 != nH2:
    #                 logger.warning(f"Error in loading {k} due to different number of heads")
    #             else:
    #                 if L1 != L2:
    #                     print("resizing key {} from {} * {} to {} * {}".format(k, L1, L1, L2, L2))
    #                     # bicubic interpolate relative_position_bias_table if not match
    #                     S1 = int(L1 ** 0.5)
    #                     S2 = int(L2 ** 0.5)
    #                     relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
    #                         relative_position_bias_table_pretrained.view(1, nH1, S1, S1), size=(S2, S2),
    #                         mode='bicubic')
    #                     state_dict[k] = relative_position_bias_table_pretrained_resized.view(
    #                         nH2, L2)     

    #         load_state_dict(self, state_dict, strict=False, logger=logger)

    # @torch.jit.ignore
    # def no_weight_decay(self):
    #     return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    # def train(self, mode=True):
    #     """Convert the model into training mode while keep layers freezed."""
    #     super(EfficientViT, self).train(mode)
    #     self._freeze_stages()
    #     if mode:
    #         for m in self.modules():
    #             if isinstance(m, _BatchNorm):
    #                 m.eval()

    def forward(self, x):
        x = self.patch_embed(x)
        outs = []
        x = self.blocks1(x)
        outs.append(x)
        x = self.blocks2(x)
        outs.append(x)
        x = self.blocks3(x)
        outs.append(x)
        return tuple(outs)

EfficientViT_m0 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [64, 128, 192],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m1 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 144, 192],
        'depth': [1, 2, 3],
        'num_heads': [2, 3, 3],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m2 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 192, 224],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 2],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m3 = {
        'img_size': 512,
        'patch_size': 16,
        'embed_dim': [128, 240, 320],
        'depth': [1, 2, 3],
        'num_heads': [4, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [5, 5, 5, 5],
    }

EfficientViT_m4 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [128, 256, 384],
        'depth': [1, 2, 3],
        'num_heads': [4, 4, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }

EfficientViT_m5 = {
        'img_size': 224,
        'patch_size': 16,
        'embed_dim': [192, 288, 384],
        'depth': [1, 3, 4],
        'num_heads': [3, 3, 4],
        'window_size': [7, 7, 7],
        'kernels': [7, 5, 3, 3],
    }


def EfficientViT_M0(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m0):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M1(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m1):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M2(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m2):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M3(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m3):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model
    

def EfficientViT_M4(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m4):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model


def EfficientViT_M5(pretrained=False, frozen_stages=0, distillation=False, fuse=False, pretrained_cfg=None, model_cfg=EfficientViT_m5):
    model = EfficientViT(frozen_stages=frozen_stages, distillation=distillation, pretrained=pretrained, **model_cfg)
    if fuse:
        replace_batchnorm(model)
    return model


if __name__ == '__main__':
    model = EfficientViT_M3()
    # print(model)

    inputs = torch.FloatTensor(np.random.rand(1, 3, 512, 512))
    outs = model(inputs)
    for out in outs:
        print(out.shape)