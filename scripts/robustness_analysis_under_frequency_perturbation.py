import argparse
import glob
import os
import json
import yaml

import numpy as np

import torch
from torch.fft import fftn, fftshift, ifftn, ifftshift
from torch.utils.data import DataLoader

from torchvision import transforms as T

from pytorch_lightning import seed_everything

from basicsr.archs import build_network
from basicsr.utils.options import ordered_yaml
from basicsr.utils.spectrum import mask_out
from basicsr.data.image_folder import ImageFolder

import dotenv

dotenv.load_dotenv(override=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args):
    root_dir = os.path.dirname(os.path.dirname(__file__))
    experiment_dir = os.path.join(root_dir, "experiments", args.experiment_name)

    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)

    opt_path = glob.glob(os.path.join(experiment_dir, '*.yml'))

    assert len(opt_path) == 1

    opt_path = opt_path[0]

    with open(opt_path, mode='r') as f:
        opt = yaml.load(f, Loader=ordered_yaml()[0])

    file_name_lack = f'test_robustness_on_frequency_lack_{args.divide}_{args.net}_{args.load_name}.json'
    file_name_noise = f'test_robustness_on_frequency_noise_{args.divide}_{args.net}_{args.load_name}.json'
    file_name_original = f'test_robustness_on_frequency_original_{args.divide}_{args.net}_{args.load_name}.json'

    if not os.path.isfile(os.path.join(experiment_dir, file_name_lack)) or not os.path.isfile(os.path.join(experiment_dir, file_name_noise)) or not os.path.isfile(os.path.join(experiment_dir, file_name_original)):
        if os.path.isfile(args.input):
            paths = [args.input]
        else:
            paths = sorted(glob.glob(os.path.join(args.input, '*')))

        dataset = ImageFolder(paths, T.Compose([
            T.RandomCrop(args.input_size),
            T.ToTensor()
        ]))
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        discriminator_ckpt = os.path.join(experiment_dir, 'models', f'{args.load_name}.pth')
        discriminator = build_network(opt['network_d'])

        discriminator.load_state_dict(torch.load(discriminator_ckpt)['params'])
        discriminator = discriminator.to(device)
        discriminator.eval()


    if not os.path.isfile(os.path.join(experiment_dir, file_name_lack)):
        stats_lack = [0] * (args.divide)
        for i, start in enumerate(np.arange(0, 1, 1 / args.divide)):
            seed_everything(args.seed)
            for img in dataloader:
                img = img.to(device)
                img_shifted_spectrum = fftshift(fftn(img, dim=(2, 3)), dim=(2, 3))

                masked_shifted_spectrum = mask_out(img_shifted_spectrum, start, start + 1 / args.divide, complement=True)

                image_lack = ifftn(ifftshift(masked_shifted_spectrum, dim=(2, 3)), dim=(2, 3)).real

                image_lack = torch.clamp(image_lack, 0, 1)

                # F.to_pil_image(img.cpu()).save(os.path.join(experiment_dir, f"{imgname}.{extension}"))
                # F.to_pil_image(image_noise.cpu()).save(os.path.join(experiment_dir, f"{imgname}_{start}.{extension}"))

                with torch.inference_mode():
                    output = discriminator(image_lack)

                stats_lack[i] += output.mean().squeeze().cpu().item()

            stats_lack[i] /= len(paths)

        with open(os.path.join(experiment_dir, file_name_lack), 'w') as f:
            json.dump(stats_lack, f)
    else:
        with open(os.path.join(experiment_dir, file_name_lack), 'r') as f:
            stats_lack = json.load(f)

    if not os.path.isfile(os.path.join(experiment_dir, file_name_noise)):
        stats_noise = [0] * (args.divide)
        for i, start in enumerate(np.arange(0, 1, 1 / args.divide)):
            seed_everything(args.seed)
            for img in dataloader:
                img = img.to(device)
                noise = torch.randn_like(img, device=img.device) * args.noise_std / 255

                noise_shifted_spectrum = fftshift(fftn(noise, dim=(2, 3)), dim=(2, 3))

                masked_shifted_spectrum = mask_out(noise_shifted_spectrum, start, start + 1 / args.divide)

                masked_noise = ifftn(ifftshift(masked_shifted_spectrum, dim=(2, 3)), dim=(2, 3)).real

                image_noise = torch.clamp(img + masked_noise, 0, 1)

                # F.to_pil_image(img.cpu()).save(os.path.join(experiment_dir, f"{imgname}.{extension}"))
                # F.to_pil_image(image_noise.cpu()).save(os.path.join(experiment_dir, f"{imgname}_{start}.{extension}"))

                with torch.inference_mode():
                    output = discriminator(image_noise)

                stats_noise[i] += output.mean().squeeze().cpu().item()

            stats_noise[i] /= len(paths)

        with open(os.path.join(experiment_dir, file_name_noise), 'w') as f:
            json.dump(stats_noise, f)
    else:
        with open(os.path.join(experiment_dir, file_name_noise), 'r') as f:
            stats_noise = json.load(f)

    if not os.path.isfile(os.path.join(experiment_dir, file_name_original)):
        stats_original = [0]
        seed_everything(args.seed)
        for img in dataloader:
            img = img.to(device)
            with torch.inference_mode():
                output = discriminator(img)

            stats_original[0] += output.mean().squeeze().cpu().item()

        stats_original[0] /= len(paths)

        with open(os.path.join(experiment_dir, file_name_original), 'w') as f:
            json.dump(stats_original, f)
    else:
        with open(os.path.join(experiment_dir, file_name_original), 'r') as f:
            stats_original = json.load(f)

    real = stats_original[0]
    stats_lack = [x - real for x in stats_lack]
    stats_noise = [x - real for x in stats_noise]

    mn = min(stats_lack + stats_noise)
    mx = max(stats_lack + stats_noise)

    delta = (abs(mn) + abs(mx)) / 20

    y_lim = (mn - delta) * 1.1, (mx + delta) * 1.1

    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
    plt.rc('font', family='serif')
    plt.rc('font', serif='Times')

    fig, axes = plt.subplots(2, 1, figsize=(9, 7))

    font_size = 28

    for ax, stats, title, label in zip(axes, [stats_lack, stats_noise], ['Frequency Masking', 'Frequency Noise'], [r'$d_\text{mask}$', r'$d_\text{noise}$']):
        ax.plot(range(len(stats)), stats, marker='o', color='#3333bb')

        # color = ['r' if x > 0 else 'g' for x in stats]
        color = ['gray' if ((x < 0 and x / mn < 5e-3) or (x > 0 and x / mx < 5e-3)) else ('r' if x > 0 else 'g') for x in stats]
        # color = ['#ffcc33' if x > 0 else '#339933' for x in stats]
        # height = max(abs(max(stats)), abs(min(stats)))
        # ax.bar(range(len(stats)), [height if x > 0 else -height for x in stats], color=color)
        ax.bar(range(len(stats)), [x + delta if x > 0 else x - delta for x in stats], color=color)
        ax.axhline(y=0, color='#000000', linestyle='--')

        x_ticks = [0, (len(stats) - 1) // 2, len(stats) - 1]
        x_ticklabels = [f"{(x / len(stats) + 1 / len(stats) / 2):.3f}" for x in x_ticks]

        ax.set_xlim(0 - 0.5, len(stats) - 1 + 0.5)
        xlabel = 'Normalized Frequency'
        ax.set_xlabel(xlabel, fontsize=font_size)
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, size=font_size-2)

        ax.set_ylim(y_lim)
        ax.set_ylabel(label, fontsize=font_size)
        for tick in ax.get_yticklabels():
            tick.set_fontsize(font_size-2)

        ax.set_title(title, size=font_size)

    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, f'test_robustness_on_frequency_{args.divide}_{args.net}.pdf'), dpi=300, bbox_inches='tight')

    r_1_lack = -1
    while r_1_lack + 1 < len(stats_lack) and stats_lack[r_1_lack + 1] >= 0:
        r_1_lack += 1
    r_2_lack = r_1_lack
    while r_2_lack + 1 < len(stats_lack) and stats_lack[r_2_lack + 1] <= 0:
        r_2_lack += 1

    r_1_noise = -1
    while r_1_noise + 1 < len(stats_noise) and stats_noise[r_1_noise + 1] <= 0:
        r_1_noise += 1
    r_2_noise = r_1_noise
    while r_2_noise + 1 < len(stats_noise) and stats_noise[r_2_noise + 1] >= 0:
        r_2_noise += 1

    r = {'lack': [(r_1_lack + 1) / len(stats_lack), (r_2_lack + 1) / len(stats_lack)],
        'noise': [(r_1_noise + 1) / len(stats_noise), (r_2_noise + 1) / len(stats_noise)]}

    with open(os.path.join(experiment_dir, f'test_robustness_on_frequency_r1_r2_{args.divide}_{args.net}_{args.load_name}.json'), 'w') as f:
        json.dump(r, f)


if __name__ == '__main__':
    """Run a robustness analysis for a model under frequency perturbations.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DIV2K/DIV2K_valid_HR', help='Input image or folder')
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--input_size', type=int, default=192)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--label', type=str, default='generated')
    parser.add_argument('--divide', type=int, default=20)

    parser.add_argument('--load_name', type=str, default='net_d')
    parser.add_argument('--net', type=str, default='unet')

    parser.add_argument('--noise_std', type=int, default=30)
    parser.add_argument('--seed', type=int, default=23)

    args = parser.parse_args()
    main(args)
