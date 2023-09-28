from itertools import chain
import torch
from torch import nn as nn
from torch.nn import functional as F
import math
from basicsr.utils.registry import ARCH_REGISTRY
from .arch_util import default_init_weights, make_layer, pixel_unshuffle


class ResidualDenseBlock(nn.Module):
    """Residual Dense Block.

    Used in RRDB block in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat=64, num_grow_ch=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        default_init_weights([self.conv1, self.conv2, self.conv3, self.conv4, self.conv5], 0.1)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        # Emperically, we use 0.2 to scale the residual for better performance
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block.

    Used in RRDB-Net in ESRGAN.

    Args:
        num_feat (int): Channel number of intermediate features.
        num_grow_ch (int): Channels for each growth.
    """

    def __init__(self, num_feat, num_grow_ch=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        # Emperically, we use 0.2 to scale the residual for better performance
        return out * 0.2 + x

class Upsampler(nn.Module):
    def __init__(self, num_feat, num_out_ch):
        super(Upsampler, self).__init__()
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, feat):
        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out

class LightUpsampler(nn.Module):
    def __init__(self, num_feat, num_out_ch):
        super(LightUpsampler, self).__init__()
        self.conv_up = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, feat):
        feat = self.lrelu(self.conv_up(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = self.conv_last(F.interpolate(feat, scale_factor=2, mode='nearest'))
        return out

class LightUpsampler2(nn.Module):
    def __init__(self, num_feat, num_out_ch):
        super(LightUpsampler2, self).__init__()
        self.conv_up = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, feat):
        feat = self.lrelu(self.conv_up(F.interpolate(feat, scale_factor=2, mode='nearest')))
        out = F.interpolate(feat, scale_factor=2, mode='nearest')
        return out

def generate_shift_freq(num_freq=1, division=10):
    freq_biases_u = []
    freq_biases_v = []

    left = division // 2 - num_freq
    right = (division + 1) // 2 + num_freq - 1

    v0 = -0.5
    v1 = 0.5
    r = (v1 - v0) / (2 * division)

    for x in range(left, right + 1):
        for y in range(left, right + 1):
            if (division % 2 == 1) and (x == division // 2) and (y == division // 2):
                continue
            freq_biases_u.append(v0 + r + 2 * r * x)
            freq_biases_v.append(v0 + r + 2 * r * y)
    return freq_biases_u, freq_biases_v

@ARCH_REGISTRY.register()
class PSRRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, num_freq=1, division=10, upsampler_version='v1', vanilla=False):
        super(PSRRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.freq_biases_u, self.freq_biases_v = generate_shift_freq(num_freq=num_freq, division=division)
        self.freq_biases_u = torch.nn.Parameter(torch.tensor(self.freq_biases_u))
        self.freq_biases_v = torch.nn.Parameter(torch.tensor(self.freq_biases_v))

        self.upsampler = Upsampler(num_feat=num_feat, num_out_ch=num_out_ch)
        self.upsampler_ps = Upsampler(num_feat=num_feat, num_out_ch=num_out_ch * 2 * len(self.freq_biases_u))

        self.vanilla = vanilla

    def forward(self, x, return_components=False):
        if self.scale == 2:
            feat = pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = pixel_unshuffle(x, scale=4)
        else:
            feat = x
        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        # upsample
        out = self.upsampler(feat)
        out_ps = self.upsampler_ps(feat)

        if return_components:
            components = [out.detach().clone()]

        N, C, H, W = out_ps.shape
        out_ps = out_ps.reshape(N, -1, 3, H, W)

        x = torch.arange(0, H, device=out.device)
        y = torch.arange(0, W, device=out.device)

        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        mod_map = 2 * math.pi * (self.freq_biases_u.view(-1, 1, 1) * grid_x.view(1, H, W) + self.freq_biases_v.view(-1, 1, 1) * grid_y.view(1, H, W))


        G_r, G_i = torch.chunk(out_ps, 2, dim=1)
        G_s = G_r * torch.cos(mod_map.view(1, -1, 1, H, W)) - G_i * torch.sin(mod_map.view(1, -1, 1, H, W))

        out += G_s.sum(dim=1)

        if return_components:
            for i in range(G_s.shape[1]):
                components.append(G_s[:, i].detach().clone())
            return out, components
        else:
            return out