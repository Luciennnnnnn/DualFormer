from copy import deepcopy

from basicsr.utils.registry import METRIC_REGISTRY
from .niqe import calculate_niqe
from .psnr_ssim import calculate_psnr, calculate_ssim
from .lpips import calculate_lpips
from .correlation import plcc, fitted_plcc, srcc, krcc

__all__ = ['calculate_psnr', 'calculate_ssim', 'calculate_niqe', 'calculate_lpips', 'plcc', 'fitted_plcc', 'srcc', 'krcc']


def calculate_metric(data, opt):
    """Calculate metric from data and options.

    Args:
        opt (dict): Configuration. It must contain:
            type (str): Model type.
    """
    opt = deepcopy(opt)
    metric_type = opt.pop('type')
    metric = METRIC_REGISTRY.get(metric_type)(**data, **opt)
    return metric
