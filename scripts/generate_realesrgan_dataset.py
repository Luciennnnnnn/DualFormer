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

from basicsr.data.degradations import circular_lowpass_kernel,random_mixed_kernels, random_add_gaussian_noise_pt, random_add_poisson_noise_pt
from basicsr.utils import FileClient, imfrombytes, img2tensor
from basicsr.utils import DiffJPEG, USMSharp
from basicsr.utils.img_process_util import filter2D

def main(args):
    seed_everything(args.seed)
    
    paths = sorted(glob.glob(os.path.join(args.input, '*')))

    os.makedirs(args.lr_folder, exist_ok=True)
    os.makedirs(args.hr_folder, exist_ok=True)

    opt = {
        'scale': args.scale,

        'resize_prob': [0.2, 0.7, 0.1],
        'resize_range': [0.15, 1.5],

        'blur_kernel_size': 21,
        'kernel_list': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob': 0.1,
        'blur_sigma': [0.2, 3],
        'betag_range': [0.5, 4],
        'betap_range': [1, 2],
        'gaussian_noise_prob': 0.5,
        'noise_range': [1, 30],
        'poisson_scale_range': [0.05, 3],
        'gray_noise_prob': 0.4,
        'jpeg_range': [30, 95],

        # the second degradation process
        'second_blur_prob': 0.8,
        'resize_prob2': [0.3, 0.4, 0.3],  # up, down, keep
        'resize_range2': [0.3, 1.2],
        'gaussian_noise_prob2': 0.5,
        'noise_range2': [1, 25],
        'poisson_scale_range2': [0.05, 2.5],
        'gray_noise_prob2': 0.4,
        'jpeg_range2': [30, 95],

        'blur_kernel_size2': 21,
        'kernel_list2': ['iso', 'aniso', 'generalized_iso', 'generalized_aniso', 'plateau_iso', 'plateau_aniso'],
        'kernel_prob2': [0.45, 0.25, 0.12, 0.03, 0.12, 0.03],
        'sinc_prob2': 0.1,
        'blur_sigma2': [0.2, 1.5],
        'betag_range2': [0.5, 4],
        'betap_range2': [1, 2],

        'final_sinc_prob': 0.8
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
        if np.random.uniform() < opt['sinc_prob']:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
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

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(kernel_range)
        if np.random.uniform() < opt['sinc_prob2']:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                opt['kernel_list2'],
                opt['kernel_prob2'],
                kernel_size,
                opt['blur_sigma2'],
                opt['blur_sigma2'], [-math.pi, math.pi],
                opt['betag_range2'],
                opt['betap_range2'],
                noise_range=None)

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < opt['final_sinc_prob']:
            kernel_size = random.choice(kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = pulse_tensor

        # BGR to RGB, HWC to CHW, numpy to tensor
        img_gt = img2tensor([img_gt], bgr2rgb=True, float32=True)[0]
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        data = {'gt': img_gt, 'kernel1': kernel, 'kernel2': kernel2, 'sinc_kernel': sinc_kernel}

        jpeger = DiffJPEG(differentiable=False).cuda()  # simulate JPEG compression artifacts
        usm_sharpener = USMSharp().cuda()  # do usm sharpening

        # training data synthesis
        gt = data['gt'].unsqueeze(0).cuda()

        ori_h, ori_w = gt.size()[2:4]
        gt = gt[:, :, :ori_h // args.scale * args.scale, :ori_w // args.scale * args.scale]

        gt_usm = usm_sharpener(gt)

        kernel1 = data['kernel1'].cuda()
        kernel2 = data['kernel2'].cuda()
        sinc_kernel = data['sinc_kernel'].cuda()

        ori_h, ori_w = gt.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(gt_usm, kernel1)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
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
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range'])
        out = torch.clamp(out, 0, 1)  # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = jpeger(out, quality=jpeg_p)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < opt['second_blur_prob']:
            out = filter2D(out, kernel2)
        # random resize
        updown_type = random.choices(['up', 'down', 'keep'], opt['resize_prob2'])[0]
        if updown_type == 'up':
            scale = np.random.uniform(1, opt['resize_range2'][1])
        elif updown_type == 'down':
            scale = np.random.uniform(opt['resize_range2'][0], 1)
        else:
            scale = 1
        mode = random.choice(['area', 'bilinear', 'bicubic'])
        out = F.interpolate(
            out, size=(int(ori_h / opt['scale'] * scale), int(ori_w / opt['scale'] * scale)), mode=mode)
        # add noise
        gray_noise_prob = opt['gray_noise_prob2']
        if np.random.uniform() < opt['gaussian_noise_prob2']:
            out = random_add_gaussian_noise_pt(
                out, sigma_range=opt['noise_range2'], clip=True, rounds=False, gray_prob=gray_noise_prob)
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=opt['poisson_scale_range2'],
                gray_prob=gray_noise_prob,
                clip=True,
                rounds=False)

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*opt['jpeg_range2'])
            out = torch.clamp(out, 0, 1)
            out = jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(['area', 'bilinear', 'bicubic'])
            out = F.interpolate(out, size=(ori_h // opt['scale'], ori_w // opt['scale']), mode=mode)
            out = filter2D(out, sinc_kernel)

        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.

        lq_img = to_pil_image(lq.squeeze())
        lq_img.save(os.path.join(args.lr_folder, img_name))

        gt_img = to_pil_image(gt.squeeze())
        gt_img.save(os.path.join(args.hr_folder, img_name))


if __name__ == '__main__':
    """Generate a SR dataset using Second-order Degradation model (https://arxiv.org/abs/2107.10833)
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--hr_folder', type=str)
    parser.add_argument('--lr_folder', type=str)
    parser.add_argument('--scale', type=int, default=8)
    parser.add_argument('--seed', type=int, default=233)

    args = parser.parse_args()
    main(args)