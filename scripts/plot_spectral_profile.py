import argparse
import os
import math

import torch

from basicsr.utils.spectrum_visualization import plot_mean, plot_std

from matplotlib import pyplot as plt
from matplotlib import font_manager

def main(args):
    root_dir = os.path.dirname(os.path.dirname(__file__))

    # teaser
    resolution = 64 * 4

    spectrum_states = ["spectral_analysis_G_DIV2K_train_HR_patch_size_256/spectrum_stats.pth",
                        "spectral_analysis_G_realesrnet_DIV2K_x4_patch_size_256/spectrum_stats.pth",
                        "spectral_analysis_G_realesrgan_DIV2K_x4_patch_size_256/spectrum_stats.pth"]

    labels = ["DIV2K", "Real-ESRNet", "Real-ESRGAN"]
    colors = ["#f2ab39", "#98c640", "#3abeec"]

    # comment following two lines if your system have not install latex
    plt.rc('text', usetex=True)
    plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

    plt.rc('font', family='serif')
    plt.rc('font', serif='Times')

    font_size = 18
    tick_font_size = 16
    fig, ax = plt.subplots(1, figsize=(10, 5))

    for spectrum_stats, label, color in zip(spectrum_states, labels, colors):
        stats = torch.load(os.path.join(root_dir, "experiments", spectrum_stats))

        mean, std = stats['mean'], stats['std']

        if args.mode == 'std':
            plot_std(mean, std, ax, label=label, c=color)
        else:
            plot_mean(mean, ax, label=label)

    # print(len(mean))
    ax.vlines(int(len(mean) / 10) + 4, 0, 100000, colors = "#e5081f", linestyles = "dashed")
    ax.vlines(int(len(mean) * (7 / 10)) + 1, 0, 100000, colors = "#e5081f", linestyles = "dashed")

    ax.text(5, 1e4, "Low", fontdict={'family': 'Times New Roman', 'size': font_size})
    ax.text(int(len(mean) * (3.7 / 10)), 1e4, "Middle", fontdict={'family': 'Times New Roman', 'size': font_size})
    ax.text(int(len(mean) * (8.15 / 10)), 1e4, "High", fontdict={'family': 'Times New Roman', 'size': font_size})

    # Settings for x-axis
    N = math.sqrt(2) * resolution
    fnyq = (N - 1) // 2
    # x_ticks = [0, fnyq / 2, fnyq]
    x_ticks = [fnyq * i / 10 for i in range(11)]
    x_ticklabels = ['%.1f' % (l / fnyq) for l in x_ticks]

    ax.set_xlim(0, fnyq)
    xlabel = 'Normalized Frequency'
    ax.set_xlabel(xlabel, fontdict={'family': 'Times New Roman', 'size': font_size})
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels, fontproperties='Times New Roman', size=tick_font_size)

    # Settings for y-axis
    ax.set_ylabel('Spectral density', fontdict={'family': 'Times New Roman', 'size': font_size})
    yticks_font = font_manager.FontProperties(family='Times New Roman')
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(yticks_font)
        tick.set_fontsize(tick_font_size)

    if std.isfinite().all():
        ymin = (mean-std).min()
        if ymin < 0:
            ymin = mean.min()
        y_lim = ymin * 0.1, (mean+std).max() * 1.1
    else:
        y_lim = mean.min() * 0.1, mean.max() * 1.1
    ax.set_ylim(y_lim)
    ax.set_yscale('log')

    # Legend
    fs = plt.rcParams['font.size']
    plt.rcParams.update({'font.size': fs*0.75})
    ax.legend(loc='upper right', ncol=2, columnspacing=1, prop={'family': 'Times New Roman', 'size': font_size})
    plt.rc('font', size=fs)

    os.makedirs(os.path.join(root_dir, "experiments", "spectral_profile"), exist_ok=True)
    plt.savefig(os.path.join(root_dir, "experiments", "spectral_profile", f"spectrum_profile_{args.name}.pdf"), dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    """Plot Spectral Profile.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default="std")
    parser.add_argument('--name', type=str, default='teaser')

    args = parser.parse_args()
    main(args)
