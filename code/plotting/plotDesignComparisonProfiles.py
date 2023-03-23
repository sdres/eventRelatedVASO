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

# load data
zscores = pd.read_csv('results/zscoreData.csv')

# Define two colors for each modality
VASOcmap = {'eventStim': '#7dadd9', 'blockStim': '#1f77b4'}
BOLDcmap = {'eventStim': '#ffae6f', 'blockStim': '#ff7f0e'}

# store as list to loop over
palettes = [VASOcmap, BOLDcmap]

for modality, cmap in zip(['VASO', 'BOLD'], palettes):

    fig, ax = plt.subplots(figsize=FS)

    # Limit plotting to modality, focus and visiotactile stimulation and remove
    # long TR block stimulation for fair comparisons
    tmp = zscores.loc[(zscores['modality'] == modality)
                      & (zscores['focus'] == 'v1')
                      & (zscores['contrast'] == 'visuotactile')
                      & (zscores['runType'] != 'blockStimLongTR')
    ]

    sns.lineplot(ax=ax,
                 data=tmp,
                 x='layer',
                 y='data',
                 hue='design',
                 palette=cmap,
                 linewidth=LW
                 )

    plt.ylabel(f'Z-score', fontsize=labelSize)

    plt.xlabel('Cortical depth', fontsize=labelSize)

    ax.set_xticks([1, 11], ['WM', 'CSF'], fontsize=tickLabelSize)


    plt.yticks(fontsize=tickLabelSize)

    # Set y-limits for modalities
    if modality == 'VASO':
        ticks = np.arange(0, 7, 1)
        plt.yticks(ticks, fontsize=tickLabelSize)
    if modality == 'BOLD':
        ticks = np.arange(0, 19, 3)
        plt.yticks(ticks, fontsize=tickLabelSize)

    plt.legend(loc='upper left', fontsize=legendTextSize)

    plt.savefig(f'results/Group_v1_{modality}_blocksVsEvents.png', bbox_inches='tight')

    plt.show()