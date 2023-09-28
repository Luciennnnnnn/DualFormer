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

def generate_shift_freq(freq_l, freq_r, num_shift_freq, schedule, version):
    if num_shift_freq == 1:
        freqs = [freq_l]
    else:
        if schedule == 'linear':
            freqs = [freq_l + (freq_r - freq_l) * k / (num_shift_freq - 1) for k in range(num_shift_freq)]
        elif schedule == 'log':
            freqs = [freq_l + (freq_r - freq_l) * math.log(0.2 * k + 1) / math.log(0.2 * (num_shift_freq - 1) + 1) for k in range(num_shift_freq)]

    if version == 'v1':
        freq_biases_u = [[freq, 0, -freq, freq] for freq in freqs]
        freq_biases_v = [[0, freq, freq, freq] for freq in freqs]

        freq_biases_u = list(chain(*freq_biases_u))
        freq_biases_v = list(chain(*freq_biases_v))
    elif version == 'v2':
        freq_biases_u = [[freq, -freq] for freq in freqs]
        freq_biases_v = [[freq, freq] for freq in freqs]

        freq_biases_u = list(chain(*freq_biases_u))
        freq_biases_v = list(chain(*freq_biases_v))
    elif version == 'v3':
        freq_biases_u = [freq for freq in freqs]
        freq_biases_v = [freq for freq in freqs]

    return freq_biases_u, freq_biases_v

@ARCH_REGISTRY.register()
class FARRDBNet(nn.Module):
    def __init__(self, num_in_ch, num_out_ch, scale=4, num_feat=64, num_block=23, num_grow_ch=32, num_shift_freq=1, freq_l=1/13, freq_r=1/13, schedule='linear', version="v1", upsampler_version='v1', vanilla=False):
        super(FARRDBNet, self).__init__()
        self.scale = scale
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = make_layer(RRDB, num_block, num_feat=num_feat, num_grow_ch=num_grow_ch)
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.upsampler = Upsampler(num_feat=num_feat, num_out_ch=num_out_ch)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.freq_biases_u, self.freq_biases_v = generate_shift_freq(freq_l, freq_r, num_shift_freq, schedule, version)

        if upsampler_version == 'v1':
            up = Upsampler
        elif upsampler_version == 'v2':
            up = LightUpsampler
        elif upsampler_version == 'v3':
            up = LightUpsampler2

        for freq_bias_u, freq_bias_v in zip(self.freq_biases_u, self.freq_biases_v):
            setattr(self, f'upsampler_{int(10000 * freq_bias_u)}_{int(10000 * freq_bias_v)}_r', up(num_feat=num_feat, num_out_ch=num_out_ch))
            setattr(self, f'upsampler_{int(10000 * freq_bias_u)}_{int(10000 * freq_bias_v)}_i', up(num_feat=num_feat, num_out_ch=num_out_ch))

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

        x = torch.arange(0, feat.shape[2] * self.scale, device=out.device)
        y = torch.arange(0, feat.shape[3] * self.scale, device=out.device)

        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

        if return_components:
            components = [out.detach().clone()]
            for freq_bias_u, freq_bias_v in zip(self.freq_biases_u, self.freq_biases_v):
                mod_map = 2 * math.pi * (freq_bias_u * grid_x + freq_bias_v * grid_y)
                G_r = getattr(self, f'upsampler_{int(10000 * freq_bias_u)}_{int(10000 * freq_bias_v)}_r')(feat)
                G_i = getattr(self, f'upsampler_{int(10000 * freq_bias_u)}_{int(10000 * freq_bias_v)}_i')(feat)
                if self.vanilla:
                    G_s = G_r - G_i
                else:
                    G_s = G_r * torch.cos(mod_map) - G_i * torch.sin(mod_map)
                components.append(G_s.detach().clone())
                out += G_s
            return out, components
            # components = [out.detach().clone()]
            # for freq_bias_u, freq_bias_v in zip(self.freq_biases_u, self.freq_biases_v):
            #     mod_map = 2 * math.pi * (freq_bias_u * grid_x + freq_bias_v * grid_y)
            #     G_r = getattr(self, f'upsampler_{int(10000 * freq_bias_u)}_{int(10000 * freq_bias_v)}_r')(feat)
            #     G_i = getattr(self, f'upsampler_{int(10000 * freq_bias_u)}_{int(10000 * freq_bias_v)}_i')(feat)

            #     G_s = torch.randn(*G_r.shape, dtype=torch.cfloat)
            #     G_s.real = G_r
            #     G_s.imag = G_i

            #     shift = torch.randn(*G_r.shape, dtype=torch.cfloat)
            #     shift.real = torch.cos(mod_map)
            #     shift.imag = torch.sin(mod_map)

            #     G_s *= shift

            #     components.append(G_s.detach().clone())
            #     # out += G_s
            # return components
        else:
            for freq_bias_u, freq_bias_v in zip(self.freq_biases_u, self.freq_biases_v):
                mod_map = 2 * math.pi * (freq_bias_u * grid_x + freq_bias_v * grid_y)
                G_r = getattr(self, f'upsampler_{int(10000 * freq_bias_u)}_{int(10000 * freq_bias_v)}_r')(feat)
                G_i = getattr(self, f'upsampler_{int(10000 * freq_bias_u)}_{int(10000 * freq_bias_v)}_i')(feat)
                if self.vanilla:
                    out += G_r - G_i
                else:
                    out += G_r * torch.cos(mod_map) - G_i * torch.sin(mod_map)
            return out