import numpy as np
import cv2
import torch

from basicsr.utils.img_process_util import filter2D

def additive_gaussian_noise(img, noise_scale=25., noise_scale_high=None):
    if noise_scale_high is None:
        noise_scale = noise_scale
    else:
        # noise_scale = torch.FloatTensor(1).uniform_(noise_scale, noise_scale_high)
        noise_scale = np.random.uniform(noise_scale, noise_scale_high)
    # return np.clip(img + np.random.randn(*img.shape) * noise_scale / 255., 0., 1.)
    return torch.clip(img + torch.randn(img.shape, dtype=img.dtype, device=img.device) * noise_scale / 255., 0., 1.)

def local_gaussian_noise(img, noise_scale=25., noise_scale_high=None, patch_size=64, patch_max_size=None):
    patch_shape = (img.shape[0], 1, img.shape[2], img.shape[3])
    patch = torch.zeros(patch_shape, dtype=torch.uint8, device=img.device)

    if patch_max_size is None:
        patch_width = patch_size
        patch_height = patch_size
    else:
        patch_width = np.random.randint(patch_size, patch_max_size + 1)
        patch_height = np.random.randint(patch_size, patch_max_size + 1)

    x = np.random.randint(0, patch_shape[3] - patch_width + 1)
    y = np.random.randint(0, patch_shape[2] - patch_height + 1)
    patch[:, :, y:y + patch_height, x:x + patch_width] = 1

    if noise_scale_high is None:
        noise_scale = noise_scale
    else:
        noise_scale = np.random.uniform(noise_scale, noise_scale_high)

    # noise = np.random.randn(*img.shape) * noise_scale / 255.
    noise = torch.randn(img.shape, dtype=img.dtype, device=img.device) * noise_scale / 255.

    # return np.clip(img + noise * patch, 0., 1.)
    return torch.clip(img + noise * patch, 0., 1.)

def uniform_noise(img, noise_scale=50., noise_scale_high=None):
    if noise_scale_high is None:
        noise_scale = noise_scale
    else:
        noise_scale = np.random.uniform(noise_scale, noise_scale_high)
    # return np.clip(img + np.random.uniform(-1, 1, img.shape) * noise_scale / 255., 0., 1.)
    return torch.clip(img + torch.empty(img.shape, dtype=img.dtype, device=img.device).uniform_(-1, 1) * noise_scale / 255., 0., 1.)

def mixture_noise(img, noise_scale_list=[15., 25., 50.], mixture_rate_list=[0.7, 0.2, 0.1]):
    batch, channel, height, width = img.shape
    noise = torch.zeros((batch * height * width, channel), dtype=img.dtype, device=img.device)
    perm = np.random.permutation(batch * height * width)
    rand = np.random.rand(batch * height * width)
    cumsum = np.cumsum([0] + mixture_rate_list)

    for i, noise_scale in enumerate(noise_scale_list):
        inds = (rand >= cumsum[i]) * (rand < cumsum[i + 1])
        if i == len(noise_scale_list) - 1:
            # noise[perm[inds], :] = np.random.uniform(
            #     -1, 1, (np.sum(inds), channel)) * noise_scale
            noise[perm[inds], :] = torch.empty((np.sum(inds), channel), dtype=img.dtype, device=img.device).uniform_(-1, 1) * noise_scale
        else:
            # noise[perm[inds], :] = np.random.randn(np.sum(inds), channel) * noise_scale
            noise[perm[inds], :] = torch.randn((np.sum(inds), channel), dtype=img.dtype, device=img.device) * noise_scale
    noise = noise.reshape(batch, height, width, channel).permute(0, 3, 1, 2)
    # return np.clip(img + noise / 255., 0., 1.)
    return torch.clip(img + noise / 255., 0., 1.)


def brown_gaussian_noise(img, noise_scale=25., noise_scale_high=None, kernel_size=5):
    kernel = (cv2.getGaussianKernel(kernel_size, 0) *
                    cv2.getGaussianKernel(kernel_size, 0).transpose())

    if noise_scale_high is None:
        noise_scale = noise_scale
    else:
        noise_scale = np.random.uniform(noise_scale, noise_scale_high)
    noise = torch.randn(img.shape, dtype=img.dtype, device=img.device) * noise_scale
    return torch.clip(img + (filter2D(noise, kernel=torch.tensor(kernel, dtype=img.dtype, device=img.device).unsqueeze(dim=0), padding_mode='constant') /
            np.sqrt(np.sum(kernel**2))) / 255., 0., 1.)


def additive_brown_gaussian_noise(img, noise_scale=25., noise_scale_high=None, kernel_size=5):
    kernel = (cv2.getGaussianKernel(kernel_size, 0) *
                    cv2.getGaussianKernel(kernel_size, 0).transpose())
    if noise_scale_high is None:
        noise_scale = noise_scale
    else:
        noise_scale = np.random.uniform(noise_scale, noise_scale_high)
    # noise = np.random.randn(*img.shape) * noise_scale
    noise = torch.randn(img.shape, dtype=img.dtype, device=img.device) * noise_scale
    return torch.clip(img + noise / 255.+ (filter2D(noise, kernel=torch.tensor(kernel, dtype=img.dtype, device=img.device).unsqueeze(dim=0), padding_mode='constant') /
                                            np.sqrt(np.sum(kernel**2))) / 255.,
                        0., 1.)

def multiplicative_gaussian_noise(img, multi_noise_scale=25., multi_noise_scale_high=None):
    if multi_noise_scale_high is None:
        multi_noise_scale = multi_noise_scale
    else:
        multi_noise_scale = np.random.uniform(multi_noise_scale,
                                        multi_noise_scale_high)
    noise = torch.randn(img.shape, dtype=img.dtype, device=img.device) * multi_noise_scale * img / 255.

    return torch.clip(img + noise, 0., 1.)

def additive_multiplicative_gaussian_noise(img, noise_scale=25., multi_noise_scale=25., noise_scale_high=None, multi_noise_scale_high=None):
    if multi_noise_scale_high is None:
        multi_noise_scale = multi_noise_scale
    else:
        multi_noise_scale = np.random.uniform(multi_noise_scale,
                                        multi_noise_scale_high)
    noise = torch.randn(img.shape, dtype=img.dtype, device=img.device) * multi_noise_scale * img / 255.

    if noise_scale_high is None:
        noise_scale = noise_scale
    else:
        noise_scale = np.random.uniform(noise_scale, noise_scale_high)
    noise = noise + torch.randn(img.shape, dtype=img.dtype, device=img.device) * noise_scale / 255.

    return torch.clip(img + noise, 0., 1.)

def poisson_noise(img, noise_lam=30., noise_lam_high=None):
    if noise_lam_high is None:
        noise_lam = noise_lam
    else:
        noise_lam = np.random.uniform(noise_lam, noise_lam_high)
    # img = np.random.poisson(noise_lam * img) / noise_lam
    img = torch.poisson(noise_lam * img) / noise_lam

    return torch.clip(img, 0., 1.)

def poisson_gaussian_noise(img, noise_lam=30., noise_scale=3., noise_lam_high=None, noise_scale_high=None):
    if noise_lam_high is None:
        noise_lam = noise_lam
    else:
        noise_lam = np.random.uniform(noise_lam, noise_lam_high)
    img = torch.poisson(noise_lam * img) / noise_lam

    if noise_scale_high is None:
        noise_scale = noise_scale
    else:
        noise_scale = np.random.uniform(noise_scale, noise_scale_high)
    img = img + torch.randn(img.shape, dtype=img.dtype, device=img.device) * noise_scale / 255.

    return torch.clip(img, 0., 1.)

noise_functions = [additive_gaussian_noise, local_gaussian_noise, uniform_noise, mixture_noise,
                    brown_gaussian_noise, additive_brown_gaussian_noise, multiplicative_gaussian_noise,
                    additive_multiplicative_gaussian_noise, poisson_noise, poisson_gaussian_noise]
