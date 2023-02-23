"""Plot timecourse for block-wise stimulation"""

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
TR = 1.3

palette = {'BOLD': 'tab:orange', 'VASO': 'tab:blue'}

data = pd.read_csv('results/blockData.csv')

fig, ax = plt.subplots(figsize=FS)

sns.lineplot(ax=ax, data=data.loc[(data['focus'] == 'v1')], x='x', y='data', hue='modality', palette=palette, linewidth=LW)


ax.axvspan(4, 4+(30/TR), color='grey', alpha=0.2, lw=0)
ax.set_ylabel('Signal change [%]', fontsize=labelSize)
ax.set_xlabel('Time [s]', fontsize=labelSize)
ax.legend(loc='lower center', fontsize=tickLabelSize)

values = (np.arange(-4,len(data['x'].unique())-4, 4)*TR).round(decimals=1)
spacing = np.arange(0,len(data['x'].unique()), 4)
ax.set_xticks(spacing, values, fontsize=tickLabelSize)

ax.tick_params(axis='y', labelsize=tickLabelSize)
ax.axhline(0, linestyle='--', color='white')
plt.savefig('results/blockTimecourseResults.png', bbox_inches='tight')
plt.show()