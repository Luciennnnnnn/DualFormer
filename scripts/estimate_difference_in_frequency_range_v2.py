import argparse
import os

import torch
from torch.fft import fftn, fftshift
import torch.nn.functional as F

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.data import build_dataset
from basicsr.utils.options import ordered_yaml
from basicsr.utils.spectrum import mask_out

import yaml
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):

    with open(args.dataset_opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    dataset = build_dataset(opt)
    
    rmse_freqs = []
    for model_path in ['experiments/pretrained_models/RealESRNet_x4plus.pth', 'experiments/pretrained_models/RealESRGAN_x4plus.pth']:
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'

        net_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
        net_g.load_state_dict(loadnet[keyname])
        net_g.eval()
        net_g.to(device)

        rmse_freq = 0
        for data in dataset:
            lq = data['lq'].unsqueeze(dim=0).to(device)
            gt = data['gt'].unsqueeze(dim=0).to(device)

            with torch.inference_mode():
                sr = net_g(lq)

            freq_gt = fftshift(fftn(gt, dim=(2, 3)), dim=(2, 3))
            freq_sr = fftshift(fftn(sr, dim=(2, 3)), dim=(2, 3))

            rmse_freq += torch.sqrt(F.mse_loss(freq_gt.abs(), freq_sr.abs()))

        
        rmse_freq /= len(dataset)
        rmse_freqs.append(rmse_freq)

    # plot two curves with the data in rmse_freqs[0] and rmse_freqs[1]
    fig, axes = plt.subplots(figsize=(9, 7))
    fig.plot(range(len(rmse_freqs[0])), rmse_freqs[0])
    fig.plot(range(len(rmse_freqs[1])), rmse_freqs[1])
    
    fig.legend()

    fig.savefig('estimate_difference_in_frequency_range_v2.png', dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    """Estimate difference in frequency ranges for a dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_opt', type=str, default="options/DIV2K_valid.yml")
    parser.add_argument('--model_path', type=str, default="experiments/pretrained_models/RealESRNet_x4plus.pth")
    parser.add_argument('--r_1', type=float, default=3 / 10)
    parser.add_argument('--r_2', type=float, default=8 / 10)

    args = parser.parse_args()
    main(args)
