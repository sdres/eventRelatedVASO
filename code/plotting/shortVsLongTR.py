"""Plot tSNR and zscore profiles for long vs short TR protocols"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

# Set modalities
modalities = ['BOLD', 'VASO']

# Set colors
palette = {'long TR': '#8DD3C7', 'short TR': '#FFFFB3'}

# =====================================================================================================================
# Plot tSNR
# =====================================================================================================================

# Load data
tSNRdata = pd.read_csv('results/qaLongVsShort.csv')

for i, modality in enumerate(['BOLD', 'VASO']):
    # Initialize plot
    fig, axs = plt.subplots(figsize=(5, 5))

    # Filter data for modality and focus
    tmp = tSNRdata.loc[(tSNRdata['modality'] == modality) & (tSNRdata['focus'] == 'v1')]

    sns.kdeplot(data=tmp,
                x='tSNR',
                hue='TRlength',
                linewidth=2,
                palette=palette
                )

    # =========================================================
    # Adapt legend
    old_legend = axs.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    axs.legend(handles, labels, loc='upper right', title='', fontsize=16)

    # =========================================================
    # Adapt Axes
    # Y-axis
    axs.set_yticks([])
    axs.set_ylabel('Density', fontsize=20)
    axs.set_xlim(0, 35)
    # X-axis
    axs.set_xticks([])
    axs.set_xlabel('')
    axs.set_xlabel('tSNR', fontsize=20)
    axs.tick_params(axis='x', labelsize=18)
    axs.set_xticks(range(0, 36, 5))

    # Save plot
    plt.savefig(f'results/Group_v1_tSNR_{modality}.png', bbox_inches='tight')
    plt.show()

# =====================================================================================================================
# Plot zscores
# =====================================================================================================================

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

# Load data
data = pd.read_csv('results/zscoreLongVsShort.csv')

for i, modality in enumerate(['BOLD', 'VASO']):
    # Initialize plot
    fig, axs = plt.subplots(figsize=(8, 5))

    # Filter data for modality and focus
    tmp = data.loc[(data['modality'] == modality) & (data['focus'] == 'v1')]

    sns.lineplot(data=tmp,
                 y='data',
                 x='layer',
                 hue='TRlength',
                 linewidth=2,
                 palette=palette,
                 errorbar=None
                 )

    # =========================================================
    # Adapt legend
    old_legend = axs.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    axs.legend(handles, labels, loc='upper left', title='', fontsize=16)

    # =========================================================
    # Adapt Axes
    # Y-axis
    axs.set_ylabel('Z-score', fontsize=20)
    if modality == 'BOLD':
        ticks = np.arange(0, 14, 2)
    if modality == 'VASO':
        ticks = np.arange(0, 6)

    plt.yticks(ticks, fontsize=tickLabelSize)

    # X-axis
    axs.set_xlabel('Cortical Depth', fontsize=20)
    axs.set_xticks([1, 11], ['WM', 'CSF'], fontsize=tickLabelSize)
    plt.gca().set_yticklabels(['{:02d}'.format(int(x)) for x in current_values])

    # Save plot
    plt.savefig(f'results/Group_v1_zscore_{modality}.png', bbox_inches='tight')
    plt.show()
