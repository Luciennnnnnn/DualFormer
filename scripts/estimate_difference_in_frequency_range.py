import argparse
import yaml

import torch
from torch.fft import fftn, fftshift
from torch.nn import functional as F

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.data import build_dataset
from basicsr.utils.options import ordered_yaml
from basicsr.utils.spectrum import mask_out

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main(args):
    with open(args.dataset_opt, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    dataset = build_dataset(opt)

    net_g_ckpt = args.model_path

    loadnet = torch.load(net_g_ckpt, map_location=torch.device('cpu'))

    if 'params_ema' in loadnet:
        keyname = 'params_ema'
    else:
        keyname = 'params'

    net_g = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32)
    net_g.load_state_dict(loadnet[keyname])
    net_g.eval()
    net_g.to(device)

    rmse1_freq = 0
    rmse2_freq = 0
    rmse3_freq = 0
    for data in dataset:
        lq = data['lq'].unsqueeze(dim=0).to(device)
        gt = data['gt'].unsqueeze(dim=0).to(device)

        with torch.inference_mode():
            sr = net_g(lq)

        freq_gt = fftshift(fftn(gt, dim=(2, 3)), dim=(2, 3))
        freq_sr = fftshift(fftn(sr, dim=(2, 3)), dim=(2, 3))

        freq_gt_1 = mask_out(freq_gt, 0, args.r_1)
        freq_gt_2 = mask_out(freq_gt, args.r_1, args.r_2)
        freq_gt_3 = mask_out(freq_gt, args.r_2, 1)

        freq_sr_1 = mask_out(freq_sr, 0, args.r_1)
        freq_sr_2 = mask_out(freq_sr, args.r_1, args.r_2)
        freq_sr_3 = mask_out(freq_sr, args.r_2, 1)

        rmse1_freq += torch.sqrt(F.mse_loss(freq_gt_1.abs(), freq_sr_1.abs()))
        rmse2_freq += torch.sqrt(F.mse_loss(freq_gt_2.abs(), freq_sr_2.abs()))
        rmse3_freq += torch.sqrt(F.mse_loss(freq_gt_3.abs(), freq_sr_3.abs()))

    rmse1_freq /= len(dataset)
    rmse2_freq /= len(dataset)
    rmse3_freq /= len(dataset)

    print(f"{rmse1_freq=}, {rmse2_freq=}, {rmse3_freq=}")

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
