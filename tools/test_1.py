from mmseg.registry import MODELS
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import timm
from torch.nn import init
import torch
import itertools
from timm.models.vision_transformer import trunc_normal_
from timm.models.layers import SqueezeExcite


class CSAHead(nn.Module):
    def __init__(self,
                 dropout=0.1,
                 window_size=7,
                 resolutions=[(40, 120), (40, 120), (20, 60), (10, 30)]
                 
                 ):
        super().__init__()
        in_channels = in_channels=(64, 96, 128, 640)
        channels = 64
        num_classes = 2
        self.pre_conv = ConvBN(in_channels[-1], channels, kernel_size=1)
        self.b4 = Block(dim=channels, num_heads=4, window_size=window_size, resolution=resolutions[-1])

        self.b3 = Block(dim=channels, num_heads=4, window_size=window_size, resolution=resolutions[-2])
        self.p3 = WF(in_channels[-2], channels)

        self.b2 = Block(dim=channels, num_heads=4, window_size=window_size, resolution=resolutions[-3])
        self.p2 = WF(in_channels[-3], channels)

        if self.training:
            self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
            self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
            # self.aux_head = AuxHead(decode_channels, num_classes)

        self.p1 = FeatureRefinementHead(in_channels[-4], channels)

        self.segmentation_head = nn.Sequential(ConvBNReLU(channels, channels),
                                               nn.Dropout2d(p=dropout, inplace=True),
                                               Conv(channels, num_classes, kernel_size=1))


    def forward(self, inputs):

        if self.training:
            # print('Res4 = ', (self.pre_conv(res4)).shape)
            x = self.b4(self.pre_conv(inputs[-1]))
            # print('x = ', x.shape)
            # h4 = self.up4(x)

            x = self.p3(x, inputs[-2])
            # print(x.shape)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res3
            x = self.b3(x)
            # print(x.shape)
            # h3 = self.up3(x)

            x = self.p2(x, inputs[-3])
            # print(x.shape)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res2
            x = self.b2(x)
            # print(x.shape)
            # h2 = x
            x = self.p1(x, inputs[-4])
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res1
            # print(x.shape)
            x = self.segmentation_head(x)
            x = F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

            # ah = h4 + h3 + h2
            # ah = self.aux_head(ah, h, w)

            return x
        else:
            x = self.b4(self.pre_conv(inputs[-1]))
            # print('x = ', x.shape)
            # h4 = self.up4(x)

            x = self.p3(x, inputs[-2])
            # print(x.shape)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res3
            x = self.b3(x)
            # print(x.shape)
            # h3 = self.up3(x)

            x = self.p2(x, inputs[-3])
            # print(x.shape)
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res2
            x = self.b2(x)
            # print(x.shape)
            # h2 = x
            x = self.p1(x, inputs[-4])
            # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            # x = x + res1
            # print(x.shape)
            x = self.segmentation_head(x)
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)

            # ah = h4 + h3 + h2
            # ah = self.aux_head(ah, h, w)

            return x


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels),
            nn.ReLU6()
        )


class ConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, norm_layer=nn.BatchNorm2d,
                 bias=False):
        super(ConvBN, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2),
            norm_layer(out_channels)
        )


class Conv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1, stride=1, bias=False):
        super(Conv, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, bias=bias,
                      dilation=dilation, stride=stride, padding=((stride - 1) + dilation * (kernel_size - 1)) // 2)
        )


class SeparableConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.ReLU6()
        )


class SeparableConvBN(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1,
                 norm_layer=nn.BatchNorm2d):
        super(SeparableConvBN, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            norm_layer(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class SeparableConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1):
        super(SeparableConv, self).__init__(
            nn.Conv2d(in_channels, in_channels, kernel_size, stride=stride, dilation=dilation,
                      padding=((stride - 1) + dilation * (kernel_size - 1)) // 2,
                      groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        )


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU6, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1, 1, 0, bias=True)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1, 1, 0, bias=True)
        self.drop = nn.Dropout(drop, inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class MobileViTv2Attention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(MobileViTv2Attention, self).__init__()
        self.fc_i = nn.Linear(d_model, 1)
        self.fc_k = nn.Linear(d_model, d_model)
        self.fc_v = nn.Linear(d_model, d_model)
        self.fc_o = nn.Linear(d_model, d_model)

        self.d_model = d_model
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, input):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :return:
        '''
        i = self.fc_i(input)  # (bs,nq,1)
        weight_i = torch.softmax(i, dim=1)  # bs,nq,1
        context_score = weight_i * self.fc_k(input)  # bs,nq,d_model
        context_vector = torch.sum(context_score, dim=1, keepdim=True)  # bs,1,d_model
        v = self.fc_v(input) * context_vector  # bs,nq,d_model
        out = self.fc_o(v)  # bs,nq,d_model

        return out


class Block(nn.Module):
    def __init__(self, dim=256, num_heads=4, window_size=7, resolution=64):
        super().__init__()

        # self.attn = GlobalLocalAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias, window_size=window_size)
        self.attn = EfficientViTBlock('s',
                                      dim, 16, num_heads,
                                      resolution,
                                      window_size,
                                      [5, 5, 5, 5])
        # self.local1 = ConvBN(dim, dim, kernel_size=3)
        # self.local2 = ConvBN(dim, dim, kernel_size=1)

        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        # # DropPath 若x为输入的张量，其通道为[B,C,H,W]，那么drop_path的含义为在一个Batch_size中，随机有drop_prob的样本，
        # # 不经过主干，而直接由分支进行恒等映射。
        # mlp_hidden_dim = int(dim * mlp_ratio)
        # self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
        #                drop=drop)
        # self.norm2 = norm_layer(dim)

    def forward(self, x):
        # print((self.norm1(x)).shape)
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # x = x + self.drop_path(self.mlp(self.norm2(x)))
        # print(x.shape)
        # local = self.local1(x) + self.local2(x)
        x = self.attn(x)
        # print(x.shape)
        return x


class WF(nn.Module):
    def __init__(self, in_channels=128, decode_channels=128, eps=1e-8):
        super(WF, self).__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        # torch.nn.Parameter可以理解为类型转换函数，将一个不可训练的Tensor转换为可训练的参数
        # torch.ones()返回一个全为1的张量，形状由参数size定义。
        self.eps = eps
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

    def forward(self, x, res):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        return x


class FeatureRefinementHead(nn.Module):
    def __init__(self, in_channels=64, decode_channels=64):
        super().__init__()
        self.pre_conv = Conv(in_channels, decode_channels, kernel_size=1)

        self.weights = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.eps = 1e-8
        self.post_conv = ConvBNReLU(decode_channels, decode_channels, kernel_size=3)

        self.pa = nn.Sequential(
            nn.Conv2d(decode_channels, decode_channels, kernel_size=3, padding=1, groups=decode_channels),
            nn.Sigmoid())
        self.ca = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                Conv(decode_channels, decode_channels // 16, kernel_size=1),
                                nn.ReLU6(),
                                Conv(decode_channels // 16, decode_channels, kernel_size=1),
                                nn.Sigmoid())

        self.shortcut = ConvBN(decode_channels, decode_channels, kernel_size=1)
        self.proj = SeparableConvBN(decode_channels, decode_channels, kernel_size=3)
        self.act = nn.ReLU6()

    def forward(self, x, res):
        # x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        # print(x.shape)
        # print((self.pre_conv(res)).shape)
        weights = nn.ReLU()(self.weights)
        fuse_weights = weights / (torch.sum(weights, dim=0) + self.eps)
        x = fuse_weights[0] * self.pre_conv(res) + fuse_weights[1] * x
        x = self.post_conv(x)
        shortcut = self.shortcut(x)
        pa = self.pa(x) * x
        ca = self.ca(x) * x
        x = pa + ca
        x = self.proj(x) + shortcut
        x = self.act(x)

        return x


# class AuxHead(nn.Module):

#     def __init__(self, in_channels=64, num_classes=8):
#         super().__init__()
#         self.conv = ConvBNReLU(in_channels, in_channels)
#         self.drop = nn.Dropout(0.1)
#         self.conv_out = Conv(in_channels, num_classes, kernel_size=1)

#     def forward(self, x, h, w):
#         feat = self.conv(x)
#         feat = self.drop(feat)
#         feat = self.conv_out(feat)
#         feat = F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False)
#         return feat
class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.ln(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads, head_dim, dropout):
#         super().__init__()
#         inner_dim = heads * head_dim
#         project_out = not (heads == 1 and head_dim == dim)
#         '''如果head==1并且head_dim==dim,project_out=False,否则project_out=True'''
#
#         self.heads = heads
#         self.scale = head_dim ** -0.5
#
#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
#
#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()
#
#     def forward(self, x):
#         # print(x.shape)
#         qkv = self.to_qkv(x).chunk(3, dim=-1)
#         # print('qkv', qkv[0].shape)
#         # chunk对数组进行分组,3表示组数，dim表示在哪一个维度.这里返回的qkv为一个包含三组张量的元组
#         q, k, v = map(lambda t: rearrange(t, 'b p n (h d) -> b p h n d', h=self.heads), qkv)
#         # print(q.shape)
#         # print(k.shape)
#         # print(v.shape)
#         dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
#         attn = self.attend(dots)
#         out = torch.matmul(attn, v)
#         out = rearrange(out, 'b p h n d -> b p n (h d)')
#         # print('self.to_out', self.to_out(out).shape)
#         return self.to_out(out)


# class Transformer(nn.Module):
#     def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
#         super().__init__()
#         self.layers = nn.ModuleList([])
#         for _ in range(depth):
#             self.layers.append(nn.ModuleList([
#                 # PreNorm(dim,Attention(dim, heads=8, head_dim=64, dropout=dropout)),
#                 PreNorm(dim, MobileViTv2Attention(dim)),
#                 PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
#             ]))
#
#     def forward(self, x):
#         out = x
#         for att, ffn in self.layers:
#             out = out + att(out)
#             out = out + ffn(out)
#         return out
#
#
# class MobileViTAttention(nn.Module):
#     def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=2, depth=3, mlp_dim=1024):
#         super().__init__()
#         self.ph, self.pw = patch_size, patch_size
#         self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
#         self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)
#
#         self.trans = Transformer(dim=dim, depth=depth, heads=8, head_dim=64, mlp_dim=mlp_dim)
#
#         self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
#         self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
#
#     def forward(self, x):
#         y = x.clone()  # bs,c,h,w
#         # y = x
#
#         ## Local Representation
#         y = self.conv2(self.conv1(x))  # bs,dim,h,w
#
#         ## Global Representation
#         _, _, h, w = y.shape
#         # print(y.shape)
#         y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)  # bs,h,w,dim
#         # print('y_a',y.shape)
#         y = self.trans(y)
#         # print('y_at', y.shape)
#         y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph,
#                       nw=w // self.pw)  # bs,dim,h,w
#
#         ## Fusion
#         y = self.conv3(y)  # bs,dim,h,w
#         y = torch.cat([x, y], 1)  # bs,2*dim,h,w
#         y = self.conv4(y)  # bs,c,h,w
#
#         return y


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
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = c.weight * w[:, None, None, None]
        b = bn.bias - bn.running_mean * bn.weight / \
            (bn.running_var + bn.eps) ** 0.5
        m = torch.nn.Conv2d(w.size(1) * self.c.groups, w.size(
            0), w.shape[2:], stride=self.c.stride, padding=self.c.padding, dilation=self.c.dilation,
                            groups=self.c.groups)
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


def channel_shuffle(x, groups):
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups

    # 将输入张量重塑为(batch_size, groups, channels_per_group, height, width)的形状
    x = x.view(batch_size, groups, channels_per_group, height, width)

    # 将通道维度与组维度进行交换
    x = x.transpose(1, 2).contiguous()

    # 将张量重塑回原始形状
    x = x.view(batch_size, -1, height, width)

    return x


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
        w = bn.weight / (bn.running_var + bn.eps) ** 0.5
        b = bn.bias - self.bn.running_mean * \
            self.bn.weight / (bn.running_var + bn.eps) ** 0.5
        w = l.weight * w[None, :]
        if l.bias is None:
            b = b @ self.l.weight.T
        else:
            b = (l.weight @ b[:, None]).view(-1) + self.l.bias
        m = torch.nn.Linear(w.size(1), w.size(0))
        m.weight.data.copy_(w)
        m.bias.data.copy_(b)
        return m


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
                 resolution=14,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.d = int(attn_ratio * key_dim)
        self.attn_ratio = attn_ratio
        self.resolution = resolution

        qkvs = []
        dws = []
        for i in range(num_heads):
            qkvs.append(Conv2d_BN(dim // (num_heads), self.key_dim * 2 + self.d, resolution=resolution))
            dws.append(Conv2d_BN(self.key_dim, self.key_dim, kernels[i], 1, kernels[i] // 2, groups=self.key_dim,
                                 resolution=resolution))
        self.qkvs = torch.nn.ModuleList(qkvs)
        self.dws = torch.nn.ModuleList(dws)
        self.proj = torch.nn.Sequential(torch.nn.ReLU(), Conv2d_BN(
            self.d * num_heads, dim, bn_weight_init=0, resolution=resolution))

        points = list(itertools.product(range(resolution), range(resolution)))
        # itertools.product 求多个可迭代对象的笛卡尔积
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
        # print(H, W)
        trainingab = self.attention_biases[:, self.attention_bias_idxs]
        x = channel_shuffle(x, groups=self.num_heads)
        feats_in = x.chunk(len(self.qkvs), dim=1)
        # torch.chunk(tensor, chunks, dim) 在给定的维度上对张量进行分块
        feats_out = []
        # feat = feats_in[0]
        # for i, qkv in enumerate(self.qkvs):
        #     if i > 0: # add the previous output to the input
        #         feat = feat + feats_in[i]
        #     # print(qkv)
        #     # print(feat.shape)
        #     feat = qkv(feat)
        #     q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1) # B, C/h, H, W
        #     # print(q.shape)
        #     # print(k.shape)
        #     # print(v.shape)
        #     q = self.dws[i](q)
        #     # print(q.shape)
        #     # print(k.shape)
        #     # print(v.shape)
        #     q, k, v = q.flatten(2), k.flatten(2), v.flatten(2) # B, C/h, N
        #     # print(q.shape)
        #     # print(k.shape)
        #     # print(v.shape)
        #     # print((q.transpose(-2, -1)).shape)
        #     # print(self.scale)
        #     # print('resolution', self.resolution)
        #     # print('trainingab[i].shape', trainingab[i].shape)
        #     # print('q.transpose(-2, -1) @ k', (q.transpose(-2, -1) @ k).shape)

        #     attn = (
        #         (q.transpose(-2, -1) @ k) * self.scale
        #         +
        #         (trainingab[i] if self.training else self.ab[i])
        #     )
        #     attn = attn.softmax(dim=-1) # BNN
        #     feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W) # BCHW
        #     feats_out.append(feat)
        for i, qkv in enumerate(self.qkvs):
            feat = feats_in[i]
            feat = qkv(feat)
            q, k, v = feat.view(B, -1, H, W).split([self.key_dim, self.key_dim, self.d], dim=1)  # B, C/h, H, W
            q = self.dws[i](q)
            q, k, v = q.flatten(2), k.flatten(2), v.flatten(2)  # B, C/h, N
            attn = (
                    (q.transpose(-2, -1) @ k) * self.scale
                    +
                    (trainingab[i] if self.training else self.ab[i])
            )
            attn = attn.softmax(dim=-1)  # BNN
            feat = (v @ attn.transpose(-2, -1)).view(B, self.d, H, W)  # BCHW
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
                 kernels=[5, 5, 5, 5], ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.resolution = resolution
        assert window_resolution > 0, 'window_size must be greater than 0'
        self.window_resolution = window_resolution

        window_resolution = min(window_resolution, resolution[0], resolution[1])
        self.attn = CascadedGroupAttention(dim, key_dim, num_heads,
                                           attn_ratio=attn_ratio,
                                           resolution=window_resolution,
                                           kernels=kernels, )

    def forward(self, x):
        H = self.resolution[0]
        W = self.resolution[1] 
        B, C, H_, W_ = x.shape
        # Only check this for classifcation models
        assert H == H_ and W == W_, 'input feature has wrong size, expect {}, got {}'.format((H, W), (H_, W_))

        if H <= self.window_resolution and W <= self.window_resolution:
            x = self.attn(x)
        else:
            x = x.permute(0, 2, 3, 1)
            # permute函数的作用是对tensor进行转置
            pad_b = (self.window_resolution - H % self.window_resolution) % self.window_resolution
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

    def __init__(self, type='s',
                 ed=128, kd=16, nh=8,
                 resolution=14,
                 window_resolution=7,
                 kernels=[5, 5, 5, 5], ):
        super().__init__()

        self.ed = ed
        self.kd = kd
        ar = ed // (kd * nh)
        self.dw0 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn0 = Residual(FFN(ed, int(ed * 2), resolution))

        if type == 's':
            self.mixer = Residual(LocalWindowAttention(ed, kd, nh, attn_ratio=ar,
                                                       resolution=resolution, window_resolution=window_resolution,
                                                       kernels=kernels))

        self.dw1 = Residual(Conv2d_BN(ed, ed, 3, 1, 1, groups=ed, bn_weight_init=0., resolution=resolution))
        self.ffn1 = Residual(FFN(ed, int(ed * 2), resolution))

    def forward(self, x):
        # print('self.ed', self.ed)
        # print('self.kd', self.kd)
        return self.ffn1(self.dw1(self.mixer(self.ffn0(self.dw0(x)))))


class EfficientViT(torch.nn.Module):
    def __init__(self, img_size=512,
                 patch_size=16,
                 in_chans=3,
                 num_classes=6,
                 stages=['s', 's', 's'],
                 embed_dim=[128, 240, 320],
                 key_dim=[16, 16, 16],
                 depth=[1, 2, 3],
                 num_heads=[4, 3, 4],
                 window_size=[7, 7, 7],
                 kernels=[5, 5, 5, 5],
                 down_ops=[['subsample', 2], ['subsample', 2], ['']],
                 distillation=False, ):
        super().__init__()

        resolution = img_size
        # Patch embedding
        self.patch_embed = torch.nn.Sequential(Conv2d_BN(in_chans, embed_dim[0] // 8, 3, 2, 1, resolution=resolution),
                                               torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 8, embed_dim[0] // 4, 3, 2, 1,
                                                         resolution=resolution // 2), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 4, embed_dim[0] // 2, 3, 2, 1,
                                                         resolution=resolution // 4), torch.nn.ReLU(),
                                               Conv2d_BN(embed_dim[0] // 2, embed_dim[0], 3, 2, 1,
                                                         resolution=resolution // 8))

        resolution = img_size // patch_size
        attn_ratio = [embed_dim[i] / (key_dim[i] * num_heads[i]) for i in range(len(embed_dim))]
        self.blocks1 = []
        self.blocks2 = []
        self.blocks3 = []

        # Build EfficientViT blocks
        for i, (stg, ed, kd, dpth, nh, ar, wd, do) in enumerate(
                zip(stages, embed_dim, key_dim, depth, num_heads, attn_ratio, window_size, down_ops)):
            for d in range(dpth):
                eval('self.blocks' + str(i + 1)).append(EfficientViTBlock(stg, ed, kd, nh, ar, resolution, wd, kernels))
            if do[0] == 'subsample':
                # Build EfficientViT downsample block
                # ('Subsample' stride)
                blk = eval('self.blocks' + str(i + 2))
                resolution_ = (resolution - 1) // do[1] + 1
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i], embed_dim[i], 3, 1, 1, groups=embed_dim[i], resolution=resolution)),
                    Residual(FFN(embed_dim[i], int(embed_dim[i] * 2), resolution)), ))
                blk.append(PatchMerging(*embed_dim[i:i + 2], resolution))
                resolution = resolution_
                blk.append(torch.nn.Sequential(Residual(
                    Conv2d_BN(embed_dim[i + 1], embed_dim[i + 1], 3, 1, 1, groups=embed_dim[i + 1],
                              resolution=resolution)),
                    Residual(
                        FFN(embed_dim[i + 1], int(embed_dim[i + 1] * 2), resolution)), ))
        self.blocks1 = torch.nn.Sequential(*self.blocks1)
        self.blocks2 = torch.nn.Sequential(*self.blocks2)
        self.blocks3 = torch.nn.Sequential(*self.blocks3)

        # Classification head
        self.head = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()
        self.distillation = distillation
        if distillation:
            self.head_dist = BN_Linear(embed_dim[-1], num_classes) if num_classes > 0 else torch.nn.Identity()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {x for x in self.state_dict().keys() if 'attention_biases' in x}

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.blocks1(x)
        x = self.blocks2(x)
        x = self.blocks3(x)
        # x = torch.nn.functional.adaptive_avg_pool2d(x, 1).flatten(1)
        # if self.distillation:
        #     x = self.head(x), self.head_dist(x)
        #     if not self.training:
        #         x = (x[0] + x[1]) / 2
        # else:
        #     x = self.head(x)
        return x


if __name__ == '__main__':
    inputs = []
    input1 = torch.randn(1, 64, 40, 120)
    input2 = torch.randn(1, 96, 40, 120)
    input3 = torch.randn(1, 128, 20, 60)
    input4 = torch.randn(1, 640, 10, 30)
    inputs.append(input1)
    inputs.append(input2)
    inputs.append(input3)
    inputs.append(input4)
    model = CSAHead()
    out = model(inputs)
    print(out.shape)
