"""Violin plot utils adapted from https://arxiv.org/abs/2104.00054."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches


def plot_single_data(metrics, output_file, corr_type="KT", first_patch_label="Our protocol"):
    """ Plot violin plots for the given metrics. """
    data1 = []
    positions = []
    ticks = []
    metric_names = []

    def set_colors(parts):
        # Color hacks
        orange = '#ff7f0e'
        for i, pc in enumerate(parts['bodies']):
            pc.set_color(orange)
        parts['cbars'].set_color([orange])
        parts['cmaxes'].set_color([orange])
        parts['cmins'].set_color([orange])

    for i, (metric_name, samples_ours) in enumerate(metrics):
        data1.append(samples_ours)
        positions.append(len(metrics) - (i + 1))
        ticks.append(len(metrics) - (i + 1))
        metric_names.append(metric_name)

    ax1 = plt.gca()

    parts1 = ax1.violinplot(data1, positions=positions, vert=False)
    set_colors(parts1)

    ax1.set_title('Uncertainty in correlation')

    ax1.set_yticks(ticks)
    ax1.set_yticklabels(metric_names)

    plt.xlabel(f'{corr_type} Correlation Coefficient')
    plt.xlim(-0.1, 0.6)

    first_patch = mpatches.Patch(color='#ff7f0e', label=first_patch_label, alpha=0.4)
    plt.legend(handles=[first_patch], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()



def plot_data(metrics, output_file, xlabel="KT Correlation Coefficient", first_patch_label="Our protocol",
              second_patch_label="Wang et al. 2022 protocol", xlim_low=0.1, xlim_high=1.0, plot_title="",
              font_size=20):
    """ Plot violin plots for the given metrics. """
    data1 = []
    positions = []
    ticks = []
    metric_names = []
    matplotlib.rcParams.update({'font.size': font_size})

    def set_colors(parts):
        # Color hacks
        blue = '#1f77b4'
        orange = '#ff7f0e'
        for i, pc in enumerate(parts['bodies']):
            if i % 2 == 0:
                pc.set_color(orange)
        parts['cbars'].set_color([orange, blue])
        parts['cmaxes'].set_color([orange, blue])
        parts['cmins'].set_color([orange, blue])

    for i, (metric_name, samples_ours, samples_squality) in enumerate(metrics):
        data1.append(samples_squality)
        data1.append(samples_ours)
        positions.append(len(metrics) - (i + 1))
        positions.append(len(metrics) - (i + 1))
        ticks.append(len(metrics) - (i + 1))
        metric_names.append(metric_name)

    ax1 = plt.gca()

    parts1 = ax1.violinplot(data1, positions=positions, vert=False)
    set_colors(parts1)

    if plot_title:
        ax1.set_title(plot_title)

    ax1.set_yticks(ticks)
    ax1.set_yticklabels(metric_names)

    plt.xticks(np.arange(-0.2, 1.2, step=0.2))
    plt.xlabel(xlabel)
    plt.xlim(xlim_low, xlim_high)

    first_patch = mpatches.Patch(color='#1f77b4', label=first_patch_label, alpha=0.4)
    second_patch = mpatches.Patch(color='#ff7f0e', label=second_patch_label, alpha=0.4)
    # plt.legend(handles=[first_patch, second_patch])
    # plt.legend(handles=[first_patch, second_patch], loc='lower left', bbox_to_anchor=(0.02, 0.02), ncol=1, prop={'size': 15})
    # plt.legend(handles=[first_patch, second_patch], loc='upper right', bbox_to_anchor=(1.0, 1.0), ncol=1, prop={'size': 15})
    plt.legend(handles=[first_patch, second_patch], loc='upper left', bbox_to_anchor=(0.02, 1.0), ncol=1, prop={'size': 15})

    plt.tight_layout()
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()
