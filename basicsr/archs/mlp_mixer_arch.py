# MIT License

# Copyright (c) 2021 Phil Wang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


from functools import partial

from torch import nn
from torch.nn.utils.parametrizations import spectral_norm

from einops.layers.torch import Rearrange, Reduce

from basicsr.utils.registry import ARCH_REGISTRY

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear, use_spectral_norm=False):
    if use_spectral_norm:
        norm = spectral_norm
    else:
        norm = nn.Identity()

    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        norm(dense(dim, inner_dim)),
        nn.GELU(),
        nn.Dropout(dropout),
        norm(dense(inner_dim, dim)),
        nn.Dropout(dropout)
    )


@ARCH_REGISTRY.register(suffix='basicsr')
def MLPMixer(img_size, patch_size, in_chans, dim, depth, num_classes, expansion_factor = 2, expansion_factor_token = 1, dropout = 0., use_spectral_norm=False):
    image_h, image_w = pair(img_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    if use_spectral_norm:
        spec_norm = spectral_norm
    else:
        spec_norm = nn.Identity()

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        spec_norm(nn.Linear((patch_size ** 2) * in_chans, dim)),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor_token, dropout, chan_first, use_spectral_norm=use_spectral_norm)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last, use_spectral_norm=use_spectral_norm))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        spec_norm(nn.Linear(dim, num_classes))
    )