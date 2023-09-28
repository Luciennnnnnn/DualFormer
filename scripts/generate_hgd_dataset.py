import argparse
import glob
import os
import math
import random

import numpy as np
import torch
from torch.nn import functional as F

from torchvision.transforms.functional import to_pil_image

from pytorch_lightning import seed_everything

from basicsr.data.degradations import random_mixed_kernels, random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils import DiffJPEG
from basicsr.utils.img_process_util import filter2D

def main(args):
    seed_everything(args.seed)
    paths = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.lr_folder, exist_ok=True)
    os.makedirs(args.hr_folder, exist_ok=True)

    opt = {
        'scale': args.scale,

        'blur_kernel_size': 21,
        'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob': 0,
        'blur_sigma': [0.1, 3],
        'betag_range': [0.5, 4],
        'betap_range': [1, 2],
        'gaussian_noise_prob': 0.5,
        'noise_range': [1, 30],
        'poisson_scale_range': [0.05, 3],
        'gray_noise_prob': 0.4,
        'jpeg_range': [40, 95],
        }

    kernel_range = [2 * v + 1 for v in range(3, 11)]  # kernel size ranges from 7 to 21
    # TODO: kernel range is now hard-coded, should be in the configure file
    pulse_tensor = torch.zeros(21, 21).float()  # convolving with pulse tensor brings no blurry effect
    pulse_tensor[10, 10] = 1

    file_client = FileClient()
    for path in paths:
        # image = Image.open(path).convert('RGB')
        img_name = os.path.basename(path)

        img_bytes = file_client.get(path, 'gt')
        img_gt = imfrombytes(img_bytes, float32=True)

        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(kernel_range)
        kernel = random_mixed_kernels(
            opt['kernel_list'],
            opt['kernel_prob'],
            kernel_size,
            opt['blur_sigma'],
            opt['blur_sigma'], [-math.pi, math.pi],
            opt['betag_range'],
            opt['betap_range'],
            noise_range=None)
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)

        data = {'gt': img_gt, 'kernel': kernel}

        jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts

        # training data synthesis
        gt = data['gt'].unsqueeze(0).cuda()

        ori_h, ori_w = gt.size()[2:4]
        gt = gt[:, :, :ori_h // args.scale * args.scale, :ori_w // args.scale * args.scale]

        kernel = data['kernel'].cuda()

        ori_h, ori_w = gt.size()[2:4]

        # blur
        if np.random.uniform() < 0.5: # gated
            out = filter2D(gt, kernel)
        else:
            out = gt

        # dowm sample
        if opt['scale'] != 1:
            out = F.interpolate(out, scale_factor=1/opt['scale'], mode='bicubic')

        # add noise
        if np.random.uniform() < 0.5: # gated
            gray_noise_prob = opt['gray_noise_prob']
            if np.random.uniform() < opt['gaussian_noise_prob']:
                out = random_add_gaussian_noise_pt(
                    out, sigma_range=opt['noise_range'], clip=True, rounds=False, gray_prob=gray_noise_prob)
            else:
                out = random_add_poisson_noise_pt(
                    out,
                    scale_range=opt['poisson_scale_range'],
                    gray_prob=gray_noise_prob,
                    clip=True,
                    rounds=False)

        # JPEG compression
        if np.random.uniform() < 0.5: # gated
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
            out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
            out = jpeger(out, quality=jpeg_p)
            
        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        lq_img = to_pil_image(lq.squeeze())
        lq_img.save(os.path.join(args.lr_folder, img_name))

        gt_img = to_pil_image(gt.squeeze())
        gt_img.save(os.path.join(args.hr_folder, img_name))


if __name__ == '__main__':
    """Generate a SR dataset using Hard Gated Degradation model (https://arxiv.org/abs/2205.04910)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--hr_folder', type=str)
    parser.add_argument('--lr_folder', type=str)
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--seed', type=int, default=233)

    args = parser.parse_args()
    main(args)