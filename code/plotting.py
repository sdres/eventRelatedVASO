'''

Plotting figures for the manuscript

'''


import pandas as pd
import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt


from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter

plt.style.use('dark_background')

# set general styles

#########################################
########## set general styles ##########
#########################################

# define figzize
FS = (10, 7)

# define linewidth to 2
LW = 2

# define fontsize size for x- and y-labels
labelSize = 24

# define fontsize size for x- and y-ticks
tickLabelSize = 18

# define fontsize legend text

legendTextSize = 18


###########################################
########## Plot blocks vs events ##########
###########################################

# load data
zscores = pd.read_csv('../results/blocksVsEventsData.csv')

# Define two colors for each modality
VASOcmap = {
    'eventStim': '#7dadd9',
    'blockStim': '#1f77b4'}
BOLDcmap = {
    'eventStim': '#ffae6f',
    'blockStim': '#ff7f0e'}

# store as list to loop over
palettes = [VASOcmap,BOLDcmap]



for modality, cmap in zip(['VASO', 'BOLD'], palettes):

    fig, ax = plt.subplots(figsize=FS)

    # Limit plotting to modality, focus and visiotactile stimulation and remove
    # long TR block stimulation for fair comparisons
    tmp = zscores.loc[
        (zscores['modality']==modality)
        & (zscores['focus']==focus)
        & (zscores['contrast']=='visiotactile')
        & (zscores['stimType']!='blockStimLongTR')
        ]

    sns.lineplot(ax=ax,
        data=tmp,
        x='layer',
        y='data',
        hue='runType',
        palette=cmap,
        linewidth = LW
        )


    plt.ylabel(f'Z-score',
        fontsize=labelSize)

    plt.xlabel(
        'WM'
        + '                                                              '
        + 'CSF',
        fontsize = labelSize
        )

    # Remove ticks for x-axis
    plt.xticks([])


    plt.yticks(fontsize=tickLabelSize)

    # Set y-limits for modalities
    if modality == 'VASO':
        ax.set_ylim(0,6)
    if modality == 'BOLD':
        ax.set_ylim(0,18)


    plt.legend(loc = 'upper left',
        fontsize = legendTextSize
        )

    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:02d}'.format(int(x)) for x in current_values])

    plt.savefig(f'../results/Group_v1_{modality}_blocksVsEvents.png',
        bbox_inches = 'tight')

    plt.show()
