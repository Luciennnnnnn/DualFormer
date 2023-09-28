from torch import nn as nn
from torch.nn.utils import spectral_norm

from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.arch_util import *

# ------------------ VGG-Style Discriminator (current) ------------------
def discriminator_block(in_channels: int, out_channels: int, norm) -> list:
    """
    Generate a basic block of VGG-style discriminator.
    Note:
        Don't use BN layer in discriminator if use WGAN-GP.
        H => floor(H / 2), W => floor(W / 2).
    :param in_channels:
        The number of input channels.
    :param out_channels:
        The number of output channels.
    :return:
        A list of conv/activation layers.
    """
    return [
        norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True),
        norm(nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True)),
        nn.LeakyReLU(negative_slope=0.2, inplace=True)
    ]


@ARCH_REGISTRY.register()
class VggDiscriminatorYN(nn.Module):
    def __init__(self, in_channels: int = 3, use_msra_init: bool = True, pool="avg", use_spectral_norm=False):
        """
        VGG-style discriminator.
        It is used to train SRGAN/ESRGAN.
        Note:
            Don't use BN layer in discriminator if use WGAN-GP.
            The number of total parameters: 7,941,897.
            Recommended input size: 192 * 192.

        :param in_channels:
            The number of input channels.
        :param use_msra_init:
            If use MSRA initialization that is also mentioned in ESRGAN.
        """
        super(VggDiscriminatorYN, self).__init__()

        if use_spectral_norm:
            norm = spectral_norm
        else:
            norm = nn.Identity()

        layers = []
        layers.extend(discriminator_block(in_channels, 64, norm))  # 192 * 192 => 96 * 96
        layers.extend(discriminator_block(64, 128, norm))  # => 48 * 48
        layers.extend(discriminator_block(128, 256, norm))  # => 24 * 24
        layers.extend(discriminator_block(256, 512, norm))  # => 12 * 12

        if pool == 'avg':
            layers.append(nn.AdaptiveAvgPool2d(output_size=(4, 4)))  # 12 * 12 => 4 * 4
        elif pool == 'max':
            layers.append(nn.AdaptiveMaxPool2d(output_size=(4, 4)))
        else:
            raise NotImplementedError("Only support: [avg, max]")

        layers.extend([
            nn.Flatten(),
            norm(nn.Linear(512 * 4 * 4, 100)),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            norm(nn.Linear(100, 1))
        ])

        if use_msra_init:
            default_init_weights(layers)

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)