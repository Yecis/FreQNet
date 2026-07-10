import math
from copy import copy
from pathlib import Path
import warnings

from jeepney.low_level import padding
from openpyxl.styles.builtins import output
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from torch import einsum
from PIL import Image
from torch.cuda import amp
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn.modules.utils import _triple, _pair, _single
# from einops import rearrange, repeat
# from einops.layers.torch import Rearrange


# from utils.torch_utils import time_synchronized
from timm.models.layers import DropPath

from torch.nn import init, Sequential
import math
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import numbers
from einops import rearrange



#########################################3
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        # print(x.shape)
        return torch.cat(x, self.d)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out

# Frequency Domain Feed-Forward Layer
class FDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FDFFN, self).__init__()

        self.patch_size = 2

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                   groups=hidden_features, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                   groups=hidden_features, bias=bias)
        self.dwconv7x7 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                                   groups=hidden_features, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()
        self.relu7 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1,
                                     groups=hidden_features, bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=5, stride=1, padding=2,
                                     groups=hidden_features, bias=bias)
        self.dwconv7x7_1 = nn.Conv2d(hidden_features, hidden_features, kernel_size=7, stride=1, padding=3,
                                     groups=hidden_features, bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()
        self.relu7_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 3, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        out_dtype = x.dtype   # 新增：记录输入 dtype

        x = self.project_in(x)
        x3 = self.relu3(self.dwconv3x3(x))
        x5 = self.relu5(self.dwconv5x5(x))
        x7 = self.relu7(self.dwconv7x7(x))

        with torch.cuda.amp.autocast(enabled=False):
            x3_patch_fft = torch.fft.rfft2(x3.float())
            x5_patch_fft = torch.fft.rfft2(x5.float())
            x7_patch_fft = torch.fft.rfft2(x7.float())

            x1_3, x2_3, x3_3 = x3_patch_fft.chunk(3, dim=1)
            x1_5, x2_5, x3_5 = x5_patch_fft.chunk(3, dim=1)
            x1_7, x2_7, x3_7 = x7_patch_fft.chunk(3, dim=1)

            x3_patch_fft = torch.cat([x1_3, x1_5, x1_7], dim=1)
            x5_patch_fft = torch.cat([x2_3, x2_5, x2_7], dim=1)
            x7_patch_fft = torch.cat([x3_3, x3_5, x3_7], dim=1)

            x3 = torch.fft.irfft2(x3_patch_fft, s=(x3.shape[2], x3.shape[3]))
            x5 = torch.fft.irfft2(x5_patch_fft, s=(x5.shape[2], x5.shape[3]))
            x7 = torch.fft.irfft2(x7_patch_fft, s=(x7.shape[2], x7.shape[3]))

        # 新增：FFT 后转回 half/float，和卷积权重保持一致
        x3 = x3.to(out_dtype)
        x5 = x5.to(out_dtype)
        x7 = x7.to(out_dtype)

        x3 = self.relu3_1(self.dwconv3x3_1(x3))
        x5 = self.relu5_1(self.dwconv5x5_1(x5))
        x7 = self.relu7_1(self.dwconv7x7_1(x7))

        x = torch.cat([x3, x5, x7], dim=1)
        x = self.project_out(x)
        return x


# Multimodal Frequency Domain Attention
class FDCA_our(nn.Module):
    def __init__(self, dim, bias):
        super(FDCA_our, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type='WithBias')

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm(dim * 2, LayerNorm_type='WithBias')

        self.patch_size = 2

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        rgb_fea_norm = self.norm1(rgb_fea)
        ir_fea_norm = self.norm1(ir_fea)

        rgb_hidden = self.to_hidden(rgb_fea_norm)
        ir_hidden = self.to_hidden(ir_fea_norm)

        rgb_q, rgb_k, rgb_v = self.to_hidden_dw(rgb_hidden).chunk(3, dim=1)
        ir_q, ir_k, ir_v = self.to_hidden_dw(ir_hidden).chunk(3, dim=1)

        out_dtype = rgb_q.dtype

        with torch.cuda.amp.autocast(enabled=False):
            rgb_q_fft = torch.fft.rfft2(rgb_q.float())
            rgb_k_fft = torch.fft.rfft2(rgb_k.float())
            ir_q_fft = torch.fft.rfft2(ir_q.float())
            ir_k_fft = torch.fft.rfft2(ir_k.float())

            rgb_out = torch.fft.irfft2(
                rgb_q_fft * ir_k_fft,
                s=(rgb_q.shape[2], rgb_q.shape[3])
            )
            ir_out = torch.fft.irfft2(
                ir_q_fft * rgb_k_fft,
                s=(ir_q.shape[2], ir_q.shape[3])
            )

        rgb_out = rgb_out.to(out_dtype)
        ir_out = ir_out.to(out_dtype)

        rgb_out = self.norm(rgb_out)
        ir_out = self.norm(ir_out)

        rgb_output = ir_v * rgb_out
        ir_output = rgb_v * ir_out

        rgb_output = self.project_out(rgb_output)
        ir_output = self.project_out(ir_output)

        return rgb_output, ir_output



# Frequency Domain Feature Aggregation Module (FDFAM)
class FDFTM_ourv1_cross(nn.Module):
    def __init__(self, dim):
        super(FDFTM_ourv1_cross, self).__init__()
        bias = False
        LayerNorm_type = 'WithBias'
        ffn_expansion_factor = 3

        self.attn = FDCA_our(dim, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FDFFN(dim, ffn_expansion_factor, bias)

        # Concat
        self.concat = Concat(dimension=1)

        # conv1x1
        self.conv = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        rgb_fea = x[0]
        ir_fea = x[1]

        rgb_fea_out, ir_fea_out = self.attn([rgb_fea, ir_fea])
        rgb_att_out = rgb_fea + rgb_fea_out
        ir_att_out = ir_fea + ir_fea_out
        rgb_fea = rgb_att_out + self.ffn(self.norm2(rgb_att_out))
        ir_fea = ir_att_out + self.ffn(self.norm2(ir_att_out))

        #out_fea = self.add([rgb_fea, ir_fea])

        fea_cat = self.concat([rgb_fea, ir_fea])
        # out_fea = self.relu(self.conv(fea_cat))

        # return out_fea
        return fea_cat
    
from ..modules.block import CIFusion

class fusion_twov1(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.FDFTM = FDFTM_ourv1_cross(dim)
        self.CIF = CIFusion(dim)

    def forward(self, x):
        y = self.FDFTM(x)   # 输入 [rgb, ir]，输出 concat 后的 2*dim
        y = self.CIF(y)     # 输入 2*dim，输出仍是 2*dim
        return y
