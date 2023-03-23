"""Plotting profiles for design comparison"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

plt.style.use('dark_background')

# ==================================================================
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

# palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
# 'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}

VASOcmap = {'visual': '#7dadd9', 'visuotactile': '#1f77b4'}
BOLDcmap = {'visual': '#ffae6f', 'visuotactile': '#ff7f0e'}

# store as list to loop over
palettes = [BOLDcmap, VASOcmap]
for focus in ['v1', 's1']:
    for j, modality in enumerate(['BOLD', 'VASO']):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        fig.subplots_adjust(top=0.8)

        for i, layer in enumerate(['superficial', 'middle', 'deep']):

            tmp = data.loc[(data['modality'] == modality) & (data['layer'] == layer) & (data['runType'] == 'eventStimRandom') & (data['focus'] == focus)]
            sns.lineplot(ax=axes[i], data=tmp, x='volume', y='data', hue='contrast', linewidth=LW, palette=palettes[j])

            if modality == 'BOLD':
                # axes[i].set_ylim(-1,7)
                axes[i].set_yticks(range(-1, 8, 2), fontsize=tickLabelSize)
            if modality == 'VASO':
                # axes[i].set_ylim(-1,4)
                axes[i].set_yticks(range(-1, 5, 1), fontsize=tickLabelSize)

            # prepare x-ticks
            ticks = range(1, 12, 3)
            labels = (np.arange(0, 11, 3)*1.3).round(decimals=1)
            for k, label in enumerate(labels):
                if label - int(label) == 0:
                    labels[k] = int(label)

            axes[i].yaxis.set_tick_params(labelsize=tickLabelSize)
            axes[i].xaxis.set_tick_params(labelsize=tickLabelSize)

            if i == 0:
                axes[i].set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
            else:
                axes[i].set_ylabel(r'', fontsize=labelSize)
                axes[i].set_yticks([])

            # Tweak x-axis
            axes[i].set_xticks(ticks)
            axes[i].set_xticklabels(labels, fontsize=tickLabelSize)
            axes[i].set_xlabel('Time [s]', fontsize=labelSize)
            axes[i].set_title(layer, fontsize=labelSize)

            # Draw lines
            axes[i].axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0)
            axes[i].axhline(0, linestyle='--', color='white')

        axes[0].get_legend().remove()
        axes[1].get_legend().remove()
        axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=legendTextSize)

        plt.suptitle(f'{modality}', fontsize=labelSize, y=0.98)
        plt.savefig(f'results/FIR_{modality}_{focus}_visualVsVisuotactile.png', bbox_inches="tight")
        plt.show()

