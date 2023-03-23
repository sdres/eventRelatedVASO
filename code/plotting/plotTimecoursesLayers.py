"""Plot event-related timecourses across layers"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# set general styles

# =================================================================================
# set general styles

# Define figzize
FS = (8, 5)
# define linewidth to 2
LW = 2
# Define fontsize size for x- and y-labels
labelSize = 24
# Define fontsize size for x- and y-ticks
tickLabelSize = 18
# Define fontsize legend text
legendTextSize = 18

data = pd.read_csv('results/firData.csv')

palettesLayers = {'VASO': ['#1f77b4', '#7dadd9', '#c9e5ff'], 'BOLD': ['#ff7f0e', '#ffae6f', '#ffdbc2']}

for focus in ['v1', 's1']:
# for focus in ['v1']:
    for i, modality in enumerate(['BOLD', 'VASO']):

        fig, ax = plt.subplots(figsize=FS)

        if focus == 's1':
            tmp = data.loc[(data['modality'] == modality) & (data['focus'] == focus) & (data['contrast'] == 'visuotactile')]

        if focus == 'v1':
            tmp = data.loc[(data['modality'] == modality) & (data['focus'] == focus)]

        for layer in tmp['layer'].unique():
            val = np.mean(tmp.loc[(tmp['volume'] == 0) & (tmp['layer'] == layer)]['data'])
            tmp['data'].loc[(tmp['layer'] == layer)] = tmp['data'].loc[(tmp['layer'] == layer)] - val

        sns.lineplot(ax=ax, data=tmp, x="volume", y="data", hue='layer', palette=palettesLayers[modality], linewidth=2)

        yLimits = ax.get_ylim()
        if modality == 'BOLD':
            # ax.set_ylim(-2, 10)
            ax.set_yticks(range(-2, 11, 2), fontsize=18)
        if modality == 'VASO':
            ax.set_yticks(range(-1, 4, 1), fontsize=18)
                # Prepare x-ticks
        ticks = range(0, 11, 2)
        labels = (np.arange(0, 11, 2)*1.3).round(decimals=1)
        for k, label in enumerate(labels):
            if label - int(label) == 0:
                labels[k] = int(label)

        ax.yaxis.set_tick_params(labelsize=tickLabelSize)
        ax.xaxis.set_tick_params(labelsize=tickLabelSize)
        ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
        #
        # if i == 0:
        #     ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
        # else:
        #     ax.set_ylabel(r'', fontsize=labelSize)
        #     # ax.set_yticks([])

        ax.legend(loc='upper right', fontsize=tickLabelSize)

        # Tweak x-axis
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=tickLabelSize)
        ax.set_xlabel('Time [s]', fontsize=labelSize)
        ax.set_title(modality, fontsize=36)

        # Draw lines
        ax.axvspan(0, 2/1.3, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation on')
        ax.axhline(0, linestyle='--', color='white')

        plt.savefig(f'results/groupLevel_{focus}_{modality}_eventResults_withLayers.png', bbox_inches="tight")

        plt.show()

# ====================================================================================
# Plot 3 participants for supplementary material
# ====================================================================================

data = pd.read_csv('results/firData.csv')

palettesLayers = {'VASO': ['#1f77b4', '#7dadd9', '#c9e5ff'], 'BOLD': ['#ff7f0e', '#ffae6f', '#ffdbc2']}

limits = {'sub-05': [19,6],
          'sub-08': [11,6],
          'sub-14': [9,6]
          }

for sub in ['sub-05', 'sub-08', 'sub-14']:
    for i, modality in enumerate(['BOLD', 'VASO']):

        fig, ax = plt.subplots(figsize=(7, 5))

        tmp = data.loc[(data['subject'] == sub) & (data['modality'] == modality) & (data['focus'] == 'v1') & (data['contrast'] == 'visuotactile')]
        for layer in tmp['layer'].unique():
            val = np.mean(tmp.loc[(tmp['volume'] == 0) & (tmp['layer'] == layer)]['data'])
            tmp['data'].loc[(tmp['layer'] == layer)] = tmp['data'].loc[(tmp['layer'] == layer)] - val

        sns.lineplot(ax=ax, data=tmp, x="volume", y="data", hue='layer', palette=palettesLayers[modality], linewidth=LW)

        yLimits = ax.get_ylim()
        ax.set_ylim(-2, limits[sub][i])
        ax.set_yticks(range(-2, limits[sub][i]+1, 2), fontsize=18)

        # prepare x-ticks
        ticks = range(1, 12, 2)
        labels = (np.arange(0, 11, 2)*1.3).round(decimals=1)
        for k, label in enumerate(labels):
            if label - int(label) == 0:
                labels[k] = int(label)

        ax.yaxis.set_tick_params(labelsize=tickLabelSize)
        ax.xaxis.set_tick_params(labelsize=tickLabelSize)

        # if i == 0:
        ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
        # else:
            # ax.set_ylabel(r'', fontsize=labelSize)
            # ax.set_yticks([])

        ax.legend(loc='upper right', fontsize=tickLabelSize)

        # tweak x-axis
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels, fontsize=tickLabelSize)
        ax.set_xlabel('Time [s]', fontsize=labelSize)
        ax.set_title(modality, fontsize=36)

        # draw lines
        ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label='stimulation on')
        ax.axhline(0, linestyle='--', color='white')

        plt.savefig(f'results/{sub}_v1_{modality}_eventResults_withLayers.png', bbox_inches="tight")

        plt.show()
