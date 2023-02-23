"""Plot zscore profiles for block-wise stimulation"""

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

plt.style.use('dark_background')

# Set modalities
modalities = ['BOLD', 'VASO']

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

# ===================================
# Plotting block-profiles
# ===================================

# load data
zscores = pd.read_csv('results/zScoreData.csv')
# Limit plotting to modality, focus and visiotactile stimulation and remove
# long TR block stimulation for fair comparisons
tmp = zscores.loc[(zscores['runType'] == 'blockStim')
                  & (zscores['focus'] == 'v1')
                  & (zscores['contrast'] == 'visuotactile')
                  & (zscores['runType'] != 'blockStimLongTR')
]

palette = {'BOLD': 'tab:orange', 'VASO': 'tab:blue'}

fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax,
             data=tmp,
             x='layer',
             y='data',
             hue='modality',
             palette=palette,
             linewidth=LW
             )

# Adapt x-axis
plt.xlabel('Cortical depth', fontsize=labelSize)
ax.set_xticks([1, 11], ['WM', 'CSF'], fontsize=tickLabelSize)

# Adapt y-axis
plt.ylabel(f'Z-score', fontsize=labelSize)
ticks = np.arange(0, 19, 3)
plt.yticks(ticks, fontsize=tickLabelSize)

plt.legend(loc='upper left', fontsize=legendTextSize)

# Save plot
plt.savefig(f'results/Group_v1_blockProfiles.png', bbox_inches='tight')
plt.show()

# ===================================
# Plotting event-profiles
# ===================================

# load data
zscores = pd.read_csv('results/zScoreData.csv')
# for sub in ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']:

# Limit plotting to modality, focus and visiotactile stimulation and remove
# long TR block stimulation for fair comparisons
tmp = zscores.loc[(zscores['runType'].str.contains('eventStim'))
                  & (zscores['focus'] == 'v1')
                  # & (zscores['runType'] == 'eventStim')
                  & (zscores['contrast'] == 'visuotactile')
]

palette = {'BOLD': 'tab:orange', 'VASO': 'tab:blue'}

fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax,
             data=tmp,
             x='layer',
             y='data',
             hue='modality',
             palette=palette,
             linewidth=LW
             )

# Adapt x-axis
plt.xlabel('Cortical depth', fontsize=labelSize)
ax.set_xticks([1, 11], ['WM', 'CSF'], fontsize=tickLabelSize)

# Adapt y-axis
plt.ylabel(f'Z-score', fontsize=labelSize)
ticks = np.arange(0, 13, 2)
plt.yticks(ticks, fontsize=tickLabelSize)

plt.legend(loc='upper left', fontsize=legendTextSize)

# Save plot
plt.savefig(f'results/Group_v1_eventProfiles.png', bbox_inches='tight')
plt.show()
