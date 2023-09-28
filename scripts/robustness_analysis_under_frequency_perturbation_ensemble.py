import argparse
import glob
import os
import json
import yaml
import numpy as np
from PIL import Image

import torch

from torch.fft import fftn, fftshift, ifftn, ifftshift

import torchvision.transforms.functional as F

from pytorch_lightning import seed_everything

from basicsr.archs import build_network
from basicsr.utils.options import ordered_yaml
from basicsr.utils.spectrum import mask_out

import matplotlib.pyplot as plt
from matplotlib import font_manager

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

    file_name_lack_1 = f'test_robustness_on_frequency_lack_{args.divide}_{args.net}_{args.load_name}.json'
    file_name_noise_1 = f'test_robustness_on_frequency_noise_{args.divide}_{args.net}_{args.load_name}.json'
    file_name_original_1 = f'test_robustness_on_frequency_original_{args.divide}_{args.net}_{args.load_name}.json'

    file_name_lack_2 = f'test_robustness_on_frequency_lack_{args.divide}_{args.net2}_{args.load_name2}.json'
    file_name_noise_2 = f'test_robustness_on_frequency_noise_{args.divide}_{args.net2}_{args.load_name2}.json'
    file_name_original_2 = f'test_robustness_on_frequency_original_{args.divide}_{args.net2}_{args.load_name2}.json'

    file_name_lack = f'test_robustness_on_frequency_lack_{args.divide}_{args.net}_{args.load_name}+{args.net2}_{args.load_name2}.json'
    file_name_noise = f'test_robustness_on_frequency_noise_{args.divide}_{args.net}_{args.load_name}+{args.net2}_{args.load_name2}.json'
    file_name_original = f'test_robustness_on_frequency_original_{args.divide}_{args.net}_{args.load_name}+{args.net2}_{args.load_name2}.json'

    file_names_lack = [file_name_lack_1, file_name_lack_2, file_name_lack]
    file_names_noise = [file_name_noise_1, file_name_noise_2, file_name_noise]
    file_names_original = [file_name_original_1, file_name_original_2, file_name_original]

    if not all([os.path.isfile(os.path.join(experiment_dir, x)) for x in file_names_lack + file_names_noise + file_names_original]):
        paths = sorted(glob.glob(os.path.join(args.input, '*')))

        discriminator_ckpt = os.path.join(experiment_dir, 'models', f'{args.load_name}.pth')
        discriminator = build_network(opt['network_d'])

        discriminator.load_state_dict(torch.load(discriminator_ckpt)['params'])
        discriminator = discriminator.to(device)
        discriminator.eval()

        discriminator2_ckpt = os.path.join(experiment_dir, 'models', f'{args.load_name2}.pth')

        discriminator2 = build_network(opt['net_spectral_d'])

        discriminator2.load_state_dict(torch.load(discriminator2_ckpt, map_location=torch.device('cpu'))['params'])
        discriminator2 = discriminator2.to(device)
        discriminator2.eval()


    if not all([os.path.isfile(os.path.join(experiment_dir, x)) for x in file_names_lack]):
        stats_lack_1 = [0] * (args.divide)
        stats_lack_2 = [0] * (args.divide)
        stats_lack = [0] * (args.divide)
        for i, start in enumerate(np.arange(0, 1, 1 / args.divide)):
            seed_everything(args.seed)
            for _ in range(args.repeat):
                for idx, path in enumerate(paths):
                    imgname, extension = os.path.splitext(os.path.basename(path))
                    print('Testing', idx, imgname)

                    img = Image.open(path).convert('RGB')

                    img = F.to_tensor(img).to(device).unsqueeze(dim=0)

                    top = np.random.randint(img.shape[2] - args.input_size + 1)
                    left = np.random.randint(img.shape[3] - args.input_size + 1)

                    img = img[:, :, top:top+args.input_size, left:left+args.input_size]

                    img_shifted_spectrum = fftshift(fftn(img, dim=(2, 3)), dim=(2, 3))
                    masked_shifted_spectrum = mask_out(img_shifted_spectrum, start, start + 1 / args.divide, complement=True)

                    image_lack = ifftn(ifftshift(masked_shifted_spectrum, dim=(2, 3)), dim=(2, 3)).real

                    image_lack = torch.clamp(image_lack, 0, 1)

                    # F.to_pil_image(img.cpu()).save(os.path.join(experiment_dir, f"{imgname}.{extension}"))
                    # F.to_pil_image(image_noise.cpu()).save(os.path.join(experiment_dir, f"{imgname}_{start}.{extension}"))

                    with torch.inference_mode():
                        output1 = discriminator(image_lack)
                        output2 = discriminator2(image_lack)

                    stats_lack_1[i] += output1.mean().squeeze().cpu().item()
                    stats_lack_2[i] += output2.mean().squeeze().cpu().item()
                    stats_lack[i] += (output1.mean().squeeze().cpu().item() + output2.mean().squeeze().cpu().item()) / 2

            stats_lack_1[i] /= len(paths) * args.repeat
            stats_lack_2[i] /= len(paths) * args.repeat
            stats_lack[i] /= len(paths) * args.repeat

        with open(os.path.join(experiment_dir, file_name_lack_1), 'w') as f:
            json.dump(stats_lack_1, f)
        with open(os.path.join(experiment_dir, file_name_lack_2), 'w') as f:
            json.dump(stats_lack_2, f)
        with open(os.path.join(experiment_dir, file_name_lack), 'w') as f:
            json.dump(stats_lack, f)
    else:
        with open(os.path.join(experiment_dir, file_name_lack_1), 'r') as f:
            stats_lack_1 = json.load(f)
        with open(os.path.join(experiment_dir, file_name_lack_2), 'r') as f:
            stats_lack_2 = json.load(f)
        with open(os.path.join(experiment_dir, file_name_lack), 'r') as f:
            stats_lack = json.load(f)

    if not all([os.path.isfile(os.path.join(experiment_dir, x)) for x in file_names_noise]):
        stats_noise_1 = [0] * (args.divide)
        stats_noise_2 = [0] * (args.divide)
        stats_noise = [0] * (args.divide)
        for i, start in enumerate(np.arange(0, 1, 1 / args.divide)):
            seed_everything(args.seed)
            for _ in range(args.repeat):
                for idx, path in enumerate(paths):
                    imgname, extension = os.path.splitext(os.path.basename(path))
                    print('Testing', idx, imgname)

                    img = Image.open(path).convert('RGB')

                    img = F.to_tensor(img).to(device).unsqueeze(dim=0)

                    top = np.random.randint(img.shape[2] - args.input_size + 1)
                    left = np.random.randint(img.shape[3] - args.input_size + 1)

                    img = img[:, :, top:top+args.input_size, left:left+args.input_size]

                    noise = torch.randn_like(img, device=img.device) * args.noise_std / 255

                    noise_shifted_spectrum = fftshift(fftn(noise, dim=(2, 3)), dim=(2, 3))
                    masked_shifted_spectrum = mask_out(noise_shifted_spectrum, start, start + 1 / args.divide)

                    masked_noise = ifftn(ifftshift(masked_shifted_spectrum, dim=(2, 3)), dim=(2, 3)).real

                    image_noise = torch.clamp(img + masked_noise, 0, 1)

                    # F.to_pil_image(img.cpu()).save(os.path.join(experiment_dir, f"{imgname}.{extension}"))
                    # F.to_pil_image(image_noise.cpu()).save(os.path.join(experiment_dir, f"{imgname}_{start}.{extension}"))

                    with torch.inference_mode():
                        output1 = discriminator(image_noise)
                        output2 = discriminator2(image_noise)

                    stats_noise_1[i] += output1.mean().squeeze().cpu().item()
                    stats_noise_2[i] += output2.mean().squeeze().cpu().item()
                    stats_noise[i] += (output1.mean().squeeze().cpu().item() + output2.mean().squeeze().cpu().item()) / 2

            stats_noise_1[i] /= len(paths) * args.repeat
            stats_noise_2[i] /= len(paths) * args.repeat
            stats_noise[i] /= len(paths) * args.repeat

        with open(os.path.join(experiment_dir, file_name_noise_1), 'w') as f:
            json.dump(stats_noise_1, f)
        with open(os.path.join(experiment_dir, file_name_noise_2), 'w') as f:
            json.dump(stats_noise_2, f)
        with open(os.path.join(experiment_dir, file_name_noise), 'w') as f:
            json.dump(stats_noise, f)
    else:
        with open(os.path.join(experiment_dir, file_name_noise_1), 'r') as f:
            stats_noise_1 = json.load(f)
        with open(os.path.join(experiment_dir, file_name_noise_2), 'r') as f:
            stats_noise_2 = json.load(f)
        with open(os.path.join(experiment_dir, file_name_noise), 'r') as f:
            stats_noise = json.load(f)

    if not all([os.path.isfile(os.path.join(experiment_dir, x)) for x in file_names_original]):
        stats_original_1 = [0]
        stats_original_2 = [0]
        stats_original = [0]
        seed_everything(args.seed)
        for _ in range(args.repeat):
            for idx, path in enumerate(paths):
                imgname, extension = os.path.splitext(os.path.basename(path))
                print('Testing', idx, imgname)

                img = Image.open(path).convert('RGB')

                img = F.to_tensor(img).to(device).unsqueeze(dim=0)

                top = np.random.randint(img.shape[2] - args.input_size + 1)
                left = np.random.randint(img.shape[3] - args.input_size + 1)

                img = img[:, :, top:top+args.input_size, left:left+args.input_size]

                with torch.inference_mode():
                    output1 = discriminator(img)
                    output2 = discriminator2(img)

                stats_original_1[0] += output1.mean().squeeze().cpu().item()
                stats_original_2[0] += output2.mean().squeeze().cpu().item()
                stats_original[0] += (output1.mean().squeeze().cpu().item() + output2.mean().squeeze().cpu().item()) / 2

        stats_original_1[0] /= len(paths) * args.repeat
        stats_original_2[0] /= len(paths) * args.repeat
        stats_original[0] /= len(paths) * args.repeat

        with open(os.path.join(experiment_dir, file_name_original_1), 'w') as f:
            json.dump(stats_original_1, f)
        with open(os.path.join(experiment_dir, file_name_original_2), 'w') as f:
            json.dump(stats_original_2, f)
        with open(os.path.join(experiment_dir, file_name_original), 'w') as f:
            json.dump(stats_original, f)
    else:
        with open(os.path.join(experiment_dir, file_name_original_1), 'r') as f:
            stats_original_1 = json.load(f)
        with open(os.path.join(experiment_dir, file_name_original_2), 'r') as f:
            stats_original_2 = json.load(f)
        with open(os.path.join(experiment_dir, file_name_original), 'r') as f:
            stats_original = json.load(f)

    real = stats_original[0]
    real_1 = stats_original_1[0]
    real_2 = stats_original_2[0]

    stats_lack = [x - real for x in stats_lack]
    stats_noise = [x - real for x in stats_noise]

    stats_lack_1 = [x - real_1 for x in stats_lack_1]
    stats_noise_1 = [x - real_1 for x in stats_noise_1]

    stats_lack_2 = [x - real_2 for x in stats_lack_2]
    stats_noise_2 = [x - real_2 for x in stats_noise_2]

    mn = min(stats_lack + stats_noise + stats_lack_1 + stats_noise_1 + stats_lack_2 + stats_noise_2)
    mx = max(stats_lack + stats_noise + stats_lack_1 + stats_noise_1 + stats_lack_2 + stats_noise_2)

    delta = (abs(mn) + abs(mx)) / 20

    y_lim = (mn - delta) * 1.1, (mx + delta) * 1.1

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for ax, stats, title in zip(axes, [stats_lack, stats_noise], ['Frequency Mask', 'Frequency Noise']):
        ax.plot(range(len(stats)), stats, marker='o', color='#3333bb')

        # color = ['r' if x > 0 else 'g' for x in stats]
        color = ['gray' if ((x < 0 and x / mn < 1e-5) or (x > 0 and x / mx < 1e-5)) else ('r' if x > 0 else 'g') for x in stats]
        # color = ['#ffcc33' if x > 0 else '#339933' for x in stats]
        # height = max(abs(max(stats)), abs(min(stats)))
        # ax.bar(range(len(stats)), [height if x > 0 else -height for x in stats], color=color)
        ax.bar(range(len(stats)), [x + delta if x > 0 else x - delta for x in stats], color=color)
        ax.axhline(y=0, color='#000000', linestyle='--')

        x_ticks = [0, (len(stats) - 1) // 2, len(stats) - 1]
        x_ticklabels = [f"{(x / len(stats) + 1 / len(stats) / 2):.3f}" for x in x_ticks]

        ax.set_xlim(0 - 0.5, len(stats) - 1 + 0.5)
        xlabel = r'$f/f_{nyq}$'
        ax.set_xlabel(xlabel, fontdict={'family': 'Times New Roman', 'size': 16})
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, fontproperties='Times New Roman', size=14)

        ax.set_ylim(y_lim)
        ax.set_ylabel("Score", fontdict={'family': 'Times New Roman', 'size': 16})
        yticks_font = font_manager.FontProperties(family='Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(yticks_font)
            tick.set_fontsize(14)

        ax.set_title(title, fontdict={'family': 'Times New Roman', 'size': 16})

    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, f'test_robustness_on_frequency_{args.divide}_{args.net}_{args.load_name}+{args.net2}_{args.load_name2}.png'), dpi=300, bbox_inches='tight')

    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for ax, stats, title in zip(axes, [stats_lack_1, stats_noise_1], ['Frequency Mask', 'Frequency Noise']):
        ax.plot(range(len(stats)), stats, marker='o', color='#3333bb')

        # color = ['r' if x > 0 else 'g' for x in stats]
        color = ['gray' if ((x < 0 and x / mn < 1e-5) or (x > 0 and x / mx < 1e-5)) else ('r' if x > 0 else 'g') for x in stats]
        # color = ['#ffcc33' if x > 0 else '#339933' for x in stats]
        # height = max(abs(max(stats)), abs(min(stats)))
        # ax.bar(range(len(stats)), [height if x > 0 else -height for x in stats], color=color)
        ax.bar(range(len(stats)), [x + delta if x > 0 else x - delta for x in stats], color=color)
        ax.axhline(y=0, color='#000000', linestyle='--')

        x_ticks = [0, (len(stats) - 1) // 2, len(stats) - 1]
        x_ticklabels = [f"{(x / len(stats) + 1 / len(stats) / 2):.3f}" for x in x_ticks]

        ax.set_xlim(0 - 0.5, len(stats) - 1 + 0.5)
        xlabel = r'$f/f_{nyq}$'
        ax.set_xlabel(xlabel, fontdict={'family': 'Times New Roman', 'size': 16})
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, fontproperties='Times New Roman', size=14)

        ax.set_ylim(y_lim)
        ax.set_ylabel("Score", fontdict={'family': 'Times New Roman', 'size': 16})
        yticks_font = font_manager.FontProperties(family='Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(yticks_font)
            tick.set_fontsize(14)

        ax.set_title(title, fontdict={'family': 'Times New Roman', 'size': 16})

    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, f'test_robustness_on_frequency_{args.divide}_{args.net}_{args.load_name}.png'), dpi=300, bbox_inches='tight')


    fig, axes = plt.subplots(2, 1, figsize=(13, 8))

    for ax, stats, title in zip(axes, [stats_lack_2, stats_noise_2], ['Frequency Mask', 'Frequency Noise']):
        ax.plot(range(len(stats)), stats, marker='o', color='#3333bb')

        # color = ['r' if x > 0 else 'g' for x in stats]
        color = ['gray' if ((x < 0 and x / mn < 1e-5) or (x > 0 and x / mx < 1e-5)) else ('r' if x > 0 else 'g') for x in stats]
        # color = ['#ffcc33' if x > 0 else '#339933' for x in stats]
        # height = max(abs(max(stats)), abs(min(stats)))
        # ax.bar(range(len(stats)), [height if x > 0 else -height for x in stats], color=color)
        ax.bar(range(len(stats)), [x + delta if x > 0 else x - delta for x in stats], color=color)
        ax.axhline(y=0, color='#000000', linestyle='--')

        x_ticks = [0, (len(stats) - 1) // 2, len(stats) - 1]
        x_ticklabels = [f"{(x / len(stats) + 1 / len(stats) / 2):.3f}" for x in x_ticks]

        ax.set_xlim(0 - 0.5, len(stats) - 1 + 0.5)
        xlabel = r'$f/f_{nyq}$'
        ax.set_xlabel(xlabel, fontdict={'family': 'Times New Roman', 'size': 16})
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_ticklabels, fontproperties='Times New Roman', size=14)

        ax.set_ylim(y_lim)
        ax.set_ylabel("Score", fontdict={'family': 'Times New Roman', 'size': 16})
        yticks_font = font_manager.FontProperties(family='Times New Roman')
        for tick in ax.get_yticklabels():
            tick.set_fontproperties(yticks_font)
            tick.set_fontsize(14)

        ax.set_title(title, fontdict={'family': 'Times New Roman', 'size': 16})

    fig.tight_layout()
    plt.savefig(os.path.join(experiment_dir, f'test_robustness_on_frequency_{args.divide}_{args.net2}_{args.load_name2}.png'), dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    """Does a robustness analysis under frequency noise/masking for ensembled discriminator
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='datasets/DIV2K/DIV2K_valid_HR')
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--input_size', type=int, default=192)
    parser.add_argument('--experiment_name', type=str)
    parser.add_argument('--label', type=str, default='generated')
    parser.add_argument('--divide', type=int, default=20)

    parser.add_argument('--load_name', type=str, default='net_d')
    parser.add_argument('--net', type=str, default='unet')

    parser.add_argument('--load_name2', type=str, default='net_spectral_d')
    parser.add_argument('--net2', type=str, default='mlpmixer2')

    parser.add_argument('--noise_std', type=int, default=30)
    parser.add_argument('--seed', type=int, default=23)

    args = parser.parse_args()
    main(args)
