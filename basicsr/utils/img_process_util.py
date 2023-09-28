import cv2
import numpy as np
import torch
from torch.nn import functional as F


def filter2D(img, kernel, padding_mode='reflect'):
    """PyTorch version of cv2.filter2D

    Args:
        img (Tensor): (b, c, h, w)
        kernel (Tensor): (b, k, k)
    """
    k = kernel.size(-1)
    b, c, h, w = img.size()
    if k % 2 == 1:
        img = F.pad(img, (k // 2, k // 2, k // 2, k // 2), mode=padding_mode)
    else:
        raise ValueError('Wrong kernel size')

    ph, pw = img.size()[-2:]

    if kernel.size(0) == 1:
        # apply the same kernel to all batch images
        img = img.view(b * c, 1, ph, pw)
        kernel = kernel.view(1, 1, k, k)
        return F.conv2d(img, kernel, padding=0).view(b, c, h, w)
    else:
        img = img.view(1, b * c, ph, pw)
        kernel = kernel.view(b, 1, k, k).repeat(1, c, 1, 1).view(b * c, 1, k, k)
        return F.conv2d(img, kernel, groups=b * c).view(b, c, h, w)


def usm_sharp(img, weight=0.5, radius=50, threshold=10):
    """USM sharpening.

    Input image: I; Blurry image: B.
    1. sharp = I + weight * (I - B)
    2. Mask = 1 if abs(I - B) > threshold, else: 0
    3. Blur mask:
    4. Out = Mask * sharp + (1 - Mask) * I


    Args:
        img (Numpy array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Sharp weight. Default: 1.
        radius (float): Kernel size of Gaussian blur. Default: 50.
        threshold (int):
    """
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype('float32')
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)

    sharp = img + weight * residual
    sharp = np.clip(sharp, 0, 1)
    return soft_mask * sharp + (1 - soft_mask) * img


class USMSharp(torch.nn.Module):

    def __init__(self, radius=50, sigma=0):
        super(USMSharp, self).__init__()
        if radius % 2 == 0:
            radius += 1
        self.radius = radius
        kernel = cv2.getGaussianKernel(radius, sigma)
        kernel = torch.FloatTensor(np.dot(kernel, kernel.transpose())).unsqueeze_(0)
        self.register_buffer('kernel', kernel)

    def forward(self, img, weight=0.5, threshold=10):
        blur = filter2D(img, self.kernel)
        residual = img - blur

        mask = torch.abs(residual) * 255 > threshold
        mask = mask.float()
        soft_mask = filter2D(mask, self.kernel)
        sharp = img + weight * residual
        sharp = torch.clip(sharp, 0, 1)
        return soft_mask * sharp + (1 - soft_mask) * img


def patchify(input, patch_size, stride):
    if isinstance(patch_size, int):
        patch_size = (patch_size, patch_size)

    if isinstance(stride, int):
        stride = (stride, stride)

    B, C, _, _ = input.size()
    patches = input.unfold(2, patch_size[0], stride[0]).unfold(3, patch_size[1], stride[1])
    patches = patches.contiguous().view(B, C, -1, patch_size[0], patch_size[1]).permute(0, 2, 1, 3, 4).reshape(-1, C, patch_size[0], patch_size[1])

    return patches

def unpatchify(input, patch_size, stride, H, W):
    if isinstance(patch_size):
        patch_size = (patch_size, patch_size)

    if isinstance(stride):
        stride = (stride, stride)

    B, C, _, _ = input.size()
    output = input.contiguous().view(B, -1, C, patch_size[0] * patch_size[1])
    output = output.permute(0, 2, 3, 1)
    output = output.contiguous().view(B, C*patch_size*patch_size, -1)

    return F.fold(output, output_size=(H, W), kernel_size=patch_size, stride=stride)