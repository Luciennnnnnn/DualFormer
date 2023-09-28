import numpy as np
from torchvision.transforms.functional import normalize

from basicsr.utils import img2tensor
from basicsr.utils.registry import METRIC_REGISTRY

try:
    import lpips
except ImportError:
    print('Please install lpips: pip install lpips')


@METRIC_REGISTRY.register()
def calculate_lpips(img, img2, **kwargs):
    loss_fn = lpips.LPIPS(net='alex').cuda()  # RGB, normalized to [-1,1]

    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    img = img.astype(np.float32) / 255.
    img2 = img2.astype(np.float32) / 255.

    img, img2 = img2tensor([img, img2], bgr2rgb=True, float32=True)
    # norm to [-1, 1]
    normalize(img, mean, std, inplace=True)
    normalize(img2, mean, std, inplace=True)

    # calculate lpips
    lpips_val = loss_fn(img.unsqueeze(0).cuda(), img2.unsqueeze(0).cuda())[0,0,0,0].detach()

    return lpips_val