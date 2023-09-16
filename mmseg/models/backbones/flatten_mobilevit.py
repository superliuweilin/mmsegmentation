import torch.nn as nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS
from torch import nn
import torch
from einops import rearrange
from torch.nn import init
import math


@MODELS.register_module()
class MobileViT(BaseModule):
    def __init__(self, image_size=(512, 512), dims = [96, 120, 144], channels = [16, 32, 48, 48, 64, 80, 96, 384], num_classes=7, input_channel=3, depths=[2, 4, 3], expansion=4, kernel_size=3,
                 patch_size=2, sr_ration=[1, 1, 1]):
        super().__init__()
        ih, iw = image_size[0], image_size[1]
        ph, pw = patch_size, patch_size
        assert iw % pw == 0 and ih % ph == 0

        self.conv1 = conv_bn(input_channel, channels[0], kernel_size=3, stride=patch_size)
        self.mv2 = nn.ModuleList([])
        self.m_vits = nn.ModuleList([])

        self.mv2.append(MV2Block(channels[0], channels[1], 1))
        self.mv2.append(MV2Block(channels[1], channels[2], 2))
        self.mv2.append(MV2Block(channels[2], channels[3], 1))
        self.mv2.append(MV2Block(channels[2], channels[3], 1))  # x2
        self.mv2.append(MV2Block(channels[3], channels[4], 2))
        # self.mv2.append(MV2Block(channels[3], channels[4], 1))
        self.m_vits.append(MobileViTAttention(channels[4], dim=dims[0], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[0], mlp_dim=int(2 * dims[0]), img_size= image_size[0] // 8, sr_ration=sr_ration[0]))
        self.mv2.append(MV2Block(channels[4], channels[5], 2))
        self.m_vits.append(MobileViTAttention(channels[5], dim=dims[1], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[1], mlp_dim=int(4 * dims[1]), img_size=image_size[0] // 16, sr_ration=sr_ration[1]))
        self.mv2.append(MV2Block(channels[5], channels[6], 2))
        self.m_vits.append(MobileViTAttention(channels[6], dim=dims[2], kernel_size=kernel_size, patch_size=patch_size,
                                              depth=depths[2], mlp_dim=int(4 * dims[2]), img_size=image_size[0] // 32, sr_ration=sr_ration[2]))

        self.conv2 = conv_bn(channels[-2], channels[-1], kernel_size=1)
        # self.upsample = Upsampler(default_conv, scale=4, n_feat=3)
        # self.pool=nn.AvgPool2d(image_size//32,1)
        # self.fc=nn.Linear(channels[-1],num_classes,bias=False)

    def forward(self, x):
        # print(x.shape)
        # _, _, H, W = x.shape
        # print(H)
        # x = self.upsample(x)
        outs = []
        y = self.conv1(x)  #
        y = self.mv2[0](y)
        y = self.mv2[1](y)  #

        y = self.mv2[2](y)
        y = self.mv2[3](y)
        res1 = y
        outs.append(res1)
        # print(y.shape)
        y = self.mv2[4](y)  #
        # print('y=', y.shape)
        y = self.m_vits[0](y)
        # print('y_vit', y.shape)
        res2 = y
        outs.append(res2)
        y = self.mv2[5](y)  #

        y = self.m_vits[1](y)
        # print(y.shape)
        res3 = y
        outs.append(res3)


        y = self.mv2[6](y)  #
        y = self.m_vits[2](y)

        y = self.conv2(y)
        # print(y.shape)
        res4 = y
        # y=self.pool(y).view(y.shape[0],-1)
        # # print(y.shape)

        # y=self.fc(y)
        outs.append(res4)

        return outs
    
class Upsampler(nn.Sequential):
    def __init__(self, conv, scale, n_feat, bn=False, act=False, bias=True):

        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feat, 4 * n_feat, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn: m.append(nn.BatchNorm2d(n_feat))
                if act: m.append(act())
        elif scale == 3:
            m.append(conv(n_feat, 9 * n_feat, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn: m.append(nn.BatchNorm2d(n_feat))
            if act: m.append(act())
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)



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


def conv_bn(inp, oup, kernel_size=3, stride=1):
    return nn.Sequential(
        nn.Conv2d(inp, oup, kernel_size=kernel_size, stride=stride, padding=kernel_size // 2),
        nn.BatchNorm2d(oup),
        nn.SiLU()
    )


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.ln = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, h, w, **kwargs):
        # print(x.shape)
        return self.fn(self.ln(x), h, w, **kwargs)


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

    def forward(self, x, h, w):
        return self.net(x)


# class Attention(nn.Module):
#     def __init__(self, dim, heads, head_dim, dropout):
#         super().__init__()
#         inner_dim = heads * head_dim
#         project_out = not (heads == 1 and head_dim == dim)
#         '''如果head==1并且head_dim==dim,project_out=False,否则project_out=True'''

#         self.heads = heads
#         self.scale = head_dim ** -0.5

#         self.attend = nn.Softmax(dim=-1)
#         self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

#         self.to_out = nn.Sequential(
#             nn.Linear(inner_dim, dim),
#             nn.Dropout(dropout)
#         ) if project_out else nn.Identity()

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

class FocusedLinearAttention(nn.Module):
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 focusing_factor=3, kernel_size=5):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.focusing_factor = focusing_factor
        self.dwc = nn.Conv2d(in_channels=head_dim, out_channels=head_dim, kernel_size=kernel_size,
                             groups=head_dim, padding=kernel_size // 2)
        self.scale = nn.Parameter(torch.zeros(size=(1, 1, dim)))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, num_patches // (sr_ratio * sr_ratio), dim)))
        print('Linear Attention sr_ratio{} f{} kernel{}'.
              format(sr_ratio, focusing_factor, kernel_size))

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            print('x_', x_.shape)
            print('x_', self.sr(x_).shape)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            
            x_ = self.norm(x_)
            
            kv = self.kv(x_).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        else:
            kv = self.kv(x).reshape(B, -1, 2, C).permute(2, 0, 1, 3)
        k, v = kv[0], kv[1]
        print(k.shape)
        print(self.positional_encoding.shape)

        k = k + self.positional_encoding
        focusing_factor = self.focusing_factor
        kernel_function = nn.ReLU()
        scale = nn.Softplus()(self.scale)
        q = kernel_function(q) + 1e-6
        k = kernel_function(k) + 1e-6
        q = q / scale
        k = k / scale
        q_norm = q.norm(dim=-1, keepdim=True)
        k_norm = k.norm(dim=-1, keepdim=True)
        q = q ** focusing_factor
        k = k ** focusing_factor
        q = (q / q.norm(dim=-1, keepdim=True)) * q_norm
        k = (k / k.norm(dim=-1, keepdim=True)) * k_norm
        q, k, v = (rearrange(x, "b n (h c) -> (b h) n c", h=self.num_heads) for x in [q, k, v])
        i, j, c, d = q.shape[-2], k.shape[-2], k.shape[-1], v.shape[-1]

        z = 1 / (torch.einsum("b i c, b c -> b i", q, k.sum(dim=1)) + 1e-6)
        if i * j * (c + d) > c * d * (i + j):
            kv = torch.einsum("b j c, b j d -> b c d", k, v)
            x = torch.einsum("b i c, b c d, b i -> b i d", q, kv, z)
        else:
            qk = torch.einsum("b i c, b j c -> b i j", q, k)
            x = torch.einsum("b i j, b j d, b i -> b i d", qk, v, z)

        if self.sr_ratio > 1:
            v = nn.functional.interpolate(v.permute(0, 2, 1), size=x.shape[1], mode='linear').permute(0, 2, 1)
        num = int(v.shape[1] ** 0.5)
        feature_map = rearrange(v, "b (w h) c -> b c w h", w=num, h=num)
        feature_map = rearrange(self.dwc(feature_map), "b c w h -> b (w h) c")
        x = x + feature_map
        x = rearrange(x, "(b h) n c -> b n (h c)", h=self.num_heads)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, dropout=0., patch_size=7, img_size=256, sr_ration=1, focusing_facto=3):
        super().__init__()
        num_patches= (img_size // patch_size) * (img_size // patch_size)
        H = img_size
        W = img_size
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, FocusedLinearAttention(dim=dim, num_patches=num_patches, num_heads=heads, qkv_bias=False, qk_scale=None, attn_drop=0., 
                                                    proj_drop=0., sr_ratio=sr_ration, focusing_factor=focusing_facto)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout))
            ]))

    def forward(self, x, h, w):
        # print('x.shape', x.shape)
        out = x
        for att, ffn in self.layers:
            out = out + att(out, h, w)
            out = out + ffn(out, h, w)
        return out


class MobileViTAttention(nn.Module):
    def __init__(self, in_channel=3, dim=512, kernel_size=3, patch_size=7, depth=3, mlp_dim=1024, img_size=256, sr_ration=1, focusing_facto=3):
        super().__init__()
        self.imgz_size = img_size
        self.ph, self.pw = patch_size, patch_size
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)
        self.conv2 = nn.Conv2d(in_channel, dim, kernel_size=1)

        self.trans = Transformer(dim=dim, depth=depth, heads=8, mlp_dim=mlp_dim, patch_size=patch_size, img_size=img_size, 
                                 sr_ration=sr_ration, focusing_facto=focusing_facto)

        self.conv3 = nn.Conv2d(dim, in_channel, kernel_size=1)
        self.conv4 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        # print(x.shape)
        print(self.imgz_size)
        # y=x.clone() #bs,c,h,w
        y = x

        ## Local Representation
        y = self.conv2(self.conv1(x))  # bs,dim,h,w

        ## Global Representation
        _, _, h, w = y.shape
        print(y.shape)
        # y = rearrange(y, 'bs dim (nh ph) (nw pw) -> bs (ph pw) (nh nw) dim', ph=self.ph, pw=self.pw)  # bs,h,w,dim
        # y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs (ph pw nh nw) dim', ph=self.ph, pw=self.pw, nh=h // self.ph,
        #               nw=w // self.pw) 
        y =y.flatten(2).transpose(1, 2)
        print('y_a',y.shape)
        y = self.trans(y, h, w)
        # print('y_at', y.shape)
        y = rearrange(y, 'bs (ph pw) (nh nw) dim -> bs dim (nh ph) (nw pw)', ph=self.ph, pw=self.pw, nh=h // self.ph,
                      nw=w // self.pw)  # bs,dim,h,w

        ## Fusion
        y = self.conv3(y)  # bs,dim,h,w
        # y=torch.cat([x,y],1) #bs,2*dim,h,w
        # y=self.conv4(y) #bs,c,h,w

        return y


class MV2Block(nn.Module):
    def __init__(self, inp, out, stride=1, expansion=1):
        super().__init__()
        self.stride = stride
        hidden_dim = inp * expansion
        self.use_res_connection = stride == 1 and inp == out

        if expansion == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=self.stride, padding=1, groups=hidden_dim,
                          bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, out, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(inp, hidden_dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=1, padding=1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(),
                nn.Conv2d(hidden_dim, out, kernel_size=1, stride=1, bias=False),
                nn.SiLU(),
                nn.BatchNorm2d(out)
            )

    def forward(self, x):
        if (self.use_res_connection):
            out = x + self.conv(x)
        else:
            out = self.conv(x)
        return out


def mobilevit_xxs():
    dims = [60, 80, 96]
    channels = [16, 16, 24, 24, 48, 64, 80, 320]
    return MobileViT(224, dims, channels, num_classes=1000)


def mobilevit_xs():
    dims = [96, 120, 144]
    channels = [16, 32, 48, 48, 64, 80, 96, 384]
    return MobileViT((512, 512), dims, channels, num_classes=1000)


def mobilevit_s():
    dims = [144, 192, 240]
    channels = [16, 32, 64, 64, 96, 128, 160, 640]
    return MobileViT((140, 480), dims, channels, num_classes=16)


def count_paratermeters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    input = torch.randn(1, 3, 512, 512)

    ### mobilevit_xxs
    mvit_xxs = mobilevit_xs()
    outs = mvit_xxs(input)
    for out in outs:
        print(out.shape)
    
    # # ### mobilevit_xs
    # # mvit_xs=mobilevit_xs()
    # # out=mvit_xs(input)
    # # # print(out.shape)

    # ### mobilevit_s
    # mvit_s=mobilevit_s()
    # print(count_paratermeters(mvit_s))
    # out=mvit_s(input)
    # # print(out.shape)
    # attention = Attention(dim=128, heads=8, head_dim=64, dropout=0.2)
    # out = attention(input)
    # print(out.shape)
