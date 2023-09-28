import argparse
import glob
import os

import numpy as np
import cv2

import torch

import torchvision.transforms.functional as F

from pytorch_lightning import seed_everything

from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.archs.swinir_arch import SwinIR
from basicsr.utils.spectrum import get_reduced_spectrum
from basicsr.utils.spectrum_visualization import plot_reduced_spectrum

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    seed_everything(args.seed)

    root_dir = os.path.dirname(os.path.dirname(__file__))
    experiment_dir = os.path.join(root_dir, "experiments", args.experiment_name)

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    model_path = args.model_path

    if args.mode == 0:
        loadnet = torch.load(model_path, map_location=torch.device('cpu'))

        if 'params_ema' in loadnet:
            keyname = 'params_ema'
        else:
            keyname = 'params'

        if args.model == 'RRDB':
            if args.scale == 4:
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=args.scale)
        elif args.model == 'SwinIR':
            model = SwinIR(upscale=args.scale, in_chans=3, img_size=48, window_size=8, img_range=1, depths=[6, 6, 6, 6, 6, 6],
            embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        else:
            NotImplementedError("Only support: [RRDB, SwinIR]")

        model.load_state_dict(loadnet[keyname], strict=True)
        model.eval()
        model.to(device)

    paths = sorted(glob.glob(os.path.join(args.input, '*')))

    spectra = []
    
    assert args.patch_size % args.scale == 0, f'patch_size should divisible scale'

    input_resolution = args.patch_size // args.scale

    for _ in range(args.repeat):
        for idx, path in enumerate(paths):
            print('Testing', idx, os.path.basename(path))

            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

            top = np.random.randint(img.shape[0] - input_resolution + 1)
            left = np.random.randint(img.shape[1] - input_resolution + 1)

            img = img[top:top+input_resolution, left:left+input_resolution]

            img = F.to_tensor(img).unsqueeze(dim=0).to(device)

            if args.mode == 0:
                with torch.inference_mode():
                    output = model(img)
            else:
                output = img

            spec = get_reduced_spectrum(output.flatten(0, 1)).unflatten(0, (output.shape[0], output.shape[1]))

            spec = spec.mean(dim=1)     # average over channels
            spectra.append(spec.cpu())

    spectra = torch.cat(spectra)
    stats = {'mean': spectra.mean(dim=0), 'std': spectra.std(dim=0)}

    torch.save(stats, os.path.join(experiment_dir, f'spectrum_stats.pth'))

    plot_reduced_spectrum(stats, resolution=args.patch_size, file_name=os.path.join(experiment_dir, f"spectrum.png"), label=args.label)

if __name__ == '__main__':
    """Estimate spectral statistics for a model/dataset.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='inputs', help='Input image folder')
    parser.add_argument('--scale', type=int, default=4, help='The final upsampling scale of the image')
    parser.add_argument('--patch_size', type=int, default=256) # Real-ESRGAN X4
    
    parser.add_argument('--model', type=str, default="RRDB")
    parser.add_argument('--model_path', type=str, default="23333")

    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--label', type=str, default='generated')

    parser.add_argument('--mode', type=int, default=0)
    parser.add_argument('--seed', type=int, default=23)

    args = parser.parse_args()
    main(args)
