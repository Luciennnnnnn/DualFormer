import torch

from torch.fft import fftn, fftshift, rfftn

def roll_quadrants(data):
    """
    Shift low frequencies to the center of fourier transform, i.e. [-N/2, ..., +N/2] -> [0, ..., N-1]
    Args:
        data: fourier transform, (NxHxW)
    Returns:
    Shifted fourier transform.
    """
    dim = data.ndim - 1

    if dim != 2:
        raise AttributeError(f'Data must be 2d but it is {dim}d.')
    if any(s % 2 == 0 for s in data.shape[1:]):
        raise RuntimeWarning('Roll quadrants for 2d input should only be used with uneven spatial sizes.')

    # for each dimension swap left and right half
    dims = tuple(range(1, dim + 1))  # add one for batch dimension
    shifts = torch.tensor(data.shape[1:]).div(2, rounding_mode='floor')  # N/2 if N even, (N-1)/2 if N odd
    return data.roll(shifts.tolist(), dims=dims)


def batch_fft(data, normalize=False):
    """
    Compute fourier transform of batch.
    Args:
        data: input tensor, (NxHxW)
    Returns:
    Batch fourier transform of input data.
    """

    dim = data.ndim - 1  # subtract one for batch dimension
    if dim != 2:
        raise AttributeError(f'Data must be 2d but it is {dim}d.')

    dims = tuple(range(1, dim + 1))  # add one for batch dimension

    if not torch.is_complex(data):
        data = torch.complex(data, torch.zeros_like(data))
    freq = fftn(data, dim=dims, norm='ortho' if normalize else 'backward')

    return freq


def azimuthal_average(image, center=None):
    # modified to tensor inputs from https://www.astrobetter.com/blog/2010/03/03/fourier-transforms-of-images-in-python/
    """
    Calculate the azimuthally averaged radial profile.
    Requires low frequencies to be at the center of the image.
    Args:
        image: Batch of 2D images, NxHxW
        center: The [x,y] pixel coordinates used as the center. The default is
             None, which then uses the center of the image (including
             fracitonal pixels).
    Returns:
    Azimuthal average over the image around the center
    """
    # Check input shapes
    assert center is None or (len(center) == 2), f'Center has to be None or len(center)=2 ' \
                                                 f'(but it is len(center)={len(center)}.'
    # Calculate the indices from the image
    H, W = image.shape[-2:]
    h, w = torch.meshgrid(torch.arange(0, H), torch.arange(0, W))

    if center is None:
        center = torch.tensor([(w.max() - w.min()) / 2.0, (h.max() - h.min()) / 2.0])

    # Compute radius for each pixel wrt center
    r = torch.stack([w - center[0], h - center[1]]).norm(2, 0)

    # Get sorted radii
    r_sorted, ind = r.flatten().sort()
    i_sorted = image.flatten(-2, -1)[..., ind]

    # Get the integer part of the radii (bin size = 1)
    r_int = r_sorted.long()  # attribute to the smaller integer

    # Find all pixels that fall within each radial bin.
    deltar = r_int[1:] - r_int[:-1]  # Assumes all radii represented, computes bin change between subsequent radii
    rind = torch.where(deltar)[0]  # location of changed radius

    # compute number of elements in each bin
    nind = rind + 1  # number of elements = idx + 1
    nind = torch.cat([torch.tensor([0]), nind, torch.tensor([H * W])])  # add borders
    nr = nind[1:] - nind[:-1]  # number of radius bin, i.e. counter for bins belonging to each radius

    # Cumulative sum to figure out sums for each radius bin
    if H % 2 == 0:
        raise NotImplementedError('Not sure if implementation correct, please check')
        rind = torch.cat([torch.tensor([0]), rind, torch.tensor([H * W - 1])])  # add borders
    else:
        rind = torch.cat([rind, torch.tensor([H * W - 1])])  # add borders
    csim = i_sorted.cumsum(-1, dtype=torch.float64)  # integrate over all values with smaller radius
    tbin = csim[..., rind[1:]] - csim[..., rind[:-1]]
    # add mean
    tbin = torch.cat([csim[:, 0:1], tbin], 1)

    radial_prof = tbin / nr.to(tbin.device)  # normalize by counted bins

    return radial_prof

def add_window(data, window=None):
    if window is not None:
        if window == 'hann':
            h_win = torch.hann_window(data.shape[1], periodic=False, dtype=data.dtype, device=data.device)
            w_win = torch.hann_window(data.shape[2], periodic=False, dtype=data.dtype, device=data.device)
            win = torch.matmul(h_win.unsqueeze(dim=1), w_win.unsqueeze(dim=0))

            data = data * win
    return data

def get_reduced_spectrum(data, normalize=False, window=None, scale='linear'):
    if (data.ndim - 1) != 2:
        raise AttributeError(f'Data must be 2d.')
    if window is not None:
        data = add_window(data, window=window)

    freq = batch_fft(data, normalize=normalize)
    power_spec = freq.real ** 2 + freq.imag ** 2
    if scale == 'log':
        power_spec = 10 * torch.log10(power_spec + 1e-7)
    N = data.shape[1]
    if N % 2 == 0:      # duplicate value for N/2 so it is put at the end of the spectrum and is not averaged with the mean value
        N_2 = N//2
        power_spec = torch.cat([power_spec[:, :N_2+1], power_spec[:, N_2:N_2+1], power_spec[:, N_2+1:]], dim=1)
        power_spec = torch.cat([power_spec[:, :, :N_2+1], power_spec[:, :, N_2:N_2+1], power_spec[:, :, N_2+1:]], dim=2)

    power_spec = roll_quadrants(power_spec)
    reduced_power_spec = azimuthal_average(power_spec)
    return reduced_power_spec

def get_power_spectrum(data, normalize=False, window=None, scale='linear'):
    if (data.ndim - 1) != 2:
        raise AttributeError(f'Data must be 2d.')

    if window is not None:
        data = add_window(data, window=window)

    freq = fftn(data, dim=(1, 2))

    freq = fftshift(freq, dim=(1, 2))
    power_spec = freq.real ** 2 + freq.imag ** 2

    if scale == 'log':
        power_spec = 10 * torch.log10(power_spec + 1e-7)

    return power_spec

def complex_mag(x):
    mag = torch.sqrt(x.real ** 2 + x.imag ** 2 + 1e-7)
    return mag

def complex_phase(x):
    phase = torch.arctan(x.imag / (x.real + 1e-7))
    return phase

def get_magnitude_spectrum(data, window=None):
    if (data.ndim - 1) != 2:
        raise AttributeError(f'Data must be 2d.')

    if window is not None:
        data = add_window(data, window=window)

    freq = fftn(data, dim=(1, 2))

    freq = fftshift(freq, dim=(1, 2))
    # mag_spec = freq.abs()
    mag_spec = complex_mag(freq)
    return mag_spec

def get_phase_spectrum(data, window=None):
    if (data.ndim - 1) != 2:
        raise AttributeError(f'Data must be 2d.')

    if window is not None:
        data = add_window(data, window=window)

    freq = fftn(data, dim=(1, 2))

    freq = fftshift(freq, dim=(1, 2))
    # phase_spec = freq.angle()
    phase_spec = complex_phase(freq)
    return phase_spec

# def get_full_spectrum(data, window=None):
#     if (data.ndim - 1) != 2:
#         raise AttributeError(f'Data must be 2d.')

#     if window is not None:
#         data = add_window(data, window=window)

#     freq = rfftn(data, dim=(1, 2))
#     mag_spec = freq.abs()
#     phase_spec = freq.angle()

#     full_spec = torch.cat([mag_spec, phase_spec], dim=-1)
#     return full_spec

import einops

def normalize(x, fix_normalize=False):
    dims = list(range(x.dim()))[1:]
    if fix_normalize:
        x = (x - x.amin(dims, keepdim=True)) / (x.amax(dims, keepdim=True) - x.amin(dims, keepdim=True) + 1e-5)
    else:
        x = (x - x.min()) / (x.max() - x.min() + 1e-5)
    return x

def spectrum_transform(x, patch_size, reduction='none', window=None, fix_normalize=False, spec_type='power', scale='linear'):
    h, w = x.shape[2] // patch_size, x.shape[3] // patch_size

    x = einops.rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size)

    if reduction == 'gray':
        x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = x.unsqueeze(dim=1)

    if spec_type == 'power':
        x = get_power_spectrum(x.flatten(0, 1), window=window, scale=scale).unflatten(0, (x.shape[0], x.shape[1]))
        if reduction == 'mean':
            x = x.mean(dim=1, keepdim=True).float()
        x = normalize(x, fix_normalize=fix_normalize)
    elif spec_type == 'magnitude':
        x = get_magnitude_spectrum(x.flatten(0, 1), window=window).unflatten(0, (x.shape[0], x.shape[1]))
        if reduction == 'mean':
            x = x.mean(dim=1, keepdim=True).float()
        x = normalize(x, fix_normalize=fix_normalize)
    elif spec_type == 'full':
        mag_spectrum = get_magnitude_spectrum(x.flatten(0, 1), window=window).unflatten(0, (x.shape[0], x.shape[1]))
        mag_spectrum = normalize(mag_spectrum, fix_normalize=fix_normalize)
        phase_spectrum = get_phase_spectrum(x.flatten(0, 1), window=window).unflatten(0, (x.shape[0], x.shape[1]))
        phase_spectrum = normalize(phase_spectrum, fix_normalize=fix_normalize)
        x = torch.cat([mag_spectrum, phase_spectrum], dim=1)
    else:
        raise NotImplementedError(f"Does not support: {spec_type}")

    x = einops.rearrange(x, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h = h, w = w)
    return x

def spectrum_transform_after_mean(x, patch_size, reduction='none', window=None, fix_normalize=False, spec_type='power'):
    h, w = x.shape[2] // patch_size, x.shape[3] // patch_size

    x = einops.rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size)

    if reduction == 'gray':
        x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = x.unsqueeze(dim=1)

    if spec_type == 'power':
        x = get_power_spectrum(x.flatten(0, 1), window=window).unflatten(0, (x.shape[0], x.shape[1]))
        x = normalize(x, fix_normalize=fix_normalize)
        if reduction == 'mean':
            x = x.mean(dim=1, keepdim=True).float()
    elif spec_type == 'magnitude':
        x = get_magnitude_spectrum(x.flatten(0, 1), window=window).unflatten(0, (x.shape[0], x.shape[1]))
        x = normalize(x, fix_normalize=fix_normalize)
        if reduction == 'mean':
            x = x.mean(dim=1, keepdim=True).float()
    elif spec_type == 'full':
        mag_spectrum = get_magnitude_spectrum(x.flatten(0, 1), window=window).unflatten(0, (x.shape[0], x.shape[1]))
        mag_spectrum = normalize(mag_spectrum, fix_normalize=fix_normalize)
        phase_spectrum = get_phase_spectrum(x.flatten(0, 1), window=window).unflatten(0, (x.shape[0], x.shape[1]))
        phase_spectrum = normalize(phase_spectrum, fix_normalize=fix_normalize)
        x = torch.cat([mag_spectrum, phase_spectrum], dim=1)
    else:
        raise NotImplementedError(f"Does not support: {spec_type}")

    x = einops.rearrange(x, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h = h, w = w)
    return x

def spectrum_transform2(x, patch_size, normalize='none'):
    h, w = x.shape[2] // patch_size, x.shape[3] // patch_size

    x = einops.rearrange(x, 'b c (h p1) (w p2) -> (b h w) c p1 p2', p1 = patch_size, p2 = patch_size)
    if normalize == 'mean':
        x1 = get_phase_spectrum(x.flatten(0, 1)).unflatten(0, (x.shape[0], x.shape[1]))
        x1 = x1.mean(dim=1, keepdim=True).float()
    elif normalize == 'gray':
        x1 = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x1 = get_phase_spectrum(x1).unsqueeze(dim=1).float()
    elif normalize == 'none':
        x1 = get_phase_spectrum(x.flatten(0, 1)).unflatten(0, (x.shape[0], x.shape[1]))
        x1 = x1.float()
    else:
        raise NotImplementedError("Only support: [mean, gray]")

    if normalize == 'mean':
        x2 = get_power_spectrum(x.flatten(0, 1)).unflatten(0, (x.shape[0], x.shape[1]))
        x2 = x2.mean(dim=1, keepdim=True).float()
    elif normalize == 'gray':
        x2 = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x2 = get_power_spectrum(x2).unsqueeze(dim=1).float()
    elif normalize == 'none':
        x2 = get_power_spectrum(x.flatten(0, 1)).unflatten(0, (x.shape[0], x.shape[1]))
        x2 = x2.float()
    else:
        raise NotImplementedError("Only support: [mean, gray]")

    x = torch.cat([x1, x2], dim=1)

    x = (x - x.min()) / (x.max() - x.min() + 1e-5)

    x = einops.rearrange(x, '(b h w) c p1 p2 -> b c (h p1) (w p2)', h = h, w = w)
    return x

def reduced_spectrum_transform(x, reduction='none', window=None, fix_normalize=False, scale='linear'):
    if reduction == 'gray':
        x = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x = x.unsqueeze(dim=1)

    x = get_reduced_spectrum(x.flatten(0, 1), window=window, scale=scale).unflatten(0, (x.shape[0], x.shape[1]))
    if reduction == 'mean':
        x = x.mean(dim=1, keepdim=True).float()
    x = normalize(x, fix_normalize=fix_normalize)
    x = x.view(x.shape[0], -1)
    return x

def mask_out(spectrum, r1, r2, complement=False):
    """Remove the frequency components out of range [r1, r2]

    Args:
        spectrum (4D Tensor): Given spectrum
        r1 (float): Left boundary
        r2 (float): Right boundary
        complement (bool): whether returns the complement

    Returns:
        4D Tensor: the spectrum only contains information in range [r1, r2]
    """
    center_x = (spectrum.shape[2] - 1) / 2
    center_y = (spectrum.shape[3] - 1) / 2

    x = torch.arange(0, spectrum.shape[2], device=spectrum.device)
    y = torch.arange(0, spectrum.shape[3], device=spectrum.device)

    grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')

    distance_from_center = torch.sqrt((grid_x - center_x)**2 + (grid_y - center_y)**2)

    r1 = distance_from_center[0][0] * r1
    r2 = distance_from_center[0][0] * r2

    mask = torch.logical_and(distance_from_center >= r1, distance_from_center < r2)

    if complement:
        mask = ~mask

    masked_spectrum = spectrum * mask

    return masked_spectrum
