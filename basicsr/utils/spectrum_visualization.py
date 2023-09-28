import math

import matplotlib.pyplot as plt


def plot_mean(mean, ax, x=None, **kwargs):
    if x is None:
        x = range(len(mean))

    ax.plot(x, mean, **kwargs)

def plot_std(mean, std, ax, x=None, **kwargs):
    if x is None:
        x = range(len(mean))

    l = ax.plot(x, mean, **kwargs)
    ax.fill_between(x, mean - std, mean + std, color=l[0]._color, alpha=0.3)

def plot_reduced_spectrum(spec, resolution, file_name, label):
    fig, ax = plt.subplots(1, figsize=(10, 6))
    mean, std = spec['mean'], spec['std']

    plot_std(mean, std, ax, c='C0', ls='--', label=label)

    # Settings for x-axis
    N = math.sqrt(2) * resolution
    fnyq = (N - 1) // 2
    x_ticks = [0, fnyq / 2, fnyq]
    x_ticklabels = ['%.1f' % (l / fnyq) for l in x_ticks]

    ax.set_xlim(0, fnyq)
    xlabel = r'$f/f_{nyq}$'
    ax.set_xlabel(xlabel)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_ticklabels)

    # Settings for y-axis
    ax.set_ylabel(r'Spectral density')
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
    ax.legend(loc='upper right', ncol=2, columnspacing=1)
    plt.rc('font', size=fs)

    plt.savefig(file_name, dpi=300, bbox_inches='tight')