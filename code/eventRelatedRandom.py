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
FS = (8, 5)

# define linewidth to 2
LW = 2

# define fontsize size for x- and y-labels
labelSize = 24

# define fontsize size for x- and y-ticks
tickLabelSize = 18

# define fontsize legend text

legendTextSize = 18

TR = 1.3


#####################################
########## Plot event FIRs ##########
#####################################

data = pd.read_csv('../results/FIRdataRandom.csv')

data = data.loc[]
# palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
# 'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}


for i, layer in enumerate(['BOLD', 'VASO']):

    fig, ax = plt.subplots(figsize=FS)

    tmp = data.loc[(data['modality']==modality)&(data['focus']=='v1')]

    sns.lineplot(ax=ax, data=tmp , x="volume", y="data", hue='layer',palette=palettesLayers[modality],linewidth=2)


    yLimits = ax.get_ylim()
    ax.set_ylim(-2,9)
    ax.set_yticks(range(-2,10,2),fontsize=18)

    # prepare x-ticks
    ticks = range(1,12,2)
    labels = (np.arange(0,11,2)*1.3).round(decimals=1)
    for k,label in enumerate(labels):
        if (label - int(label) == 0):
            labels[k] = int(label)

    ax.yaxis.set_tick_params(labelsize=tickLabelSize)
    ax.xaxis.set_tick_params(labelsize=tickLabelSize)

    if i == 0:
        ax.set_ylabel(r'Signal [$\beta$]', fontsize=labelSize)
    else:
        ax.set_ylabel(r'', fontsize=labelSize)
        ax.set_yticks([])

    ax.legend(loc='upper right',fontsize=tickLabelSize)

    # tweak x-axis
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels,
        fontsize=tickLabelSize)
    ax.set_xlabel('Time [s]',
        fontsize=labelSize)
    ax.set_title(modality, fontsize=36)

    # draw lines
    ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
    ax.axhline(0,linestyle='--',color='white')


    # ax.axvline((np.arange(0,11)*1.3).round(decimals=1)[3],linestyle='--',color='white')

    plt.savefig(f'../results/groupLevel_{focus}_{modality}_eventResults_withLayers.png', bbox_inches = "tight")


    plt.show()
