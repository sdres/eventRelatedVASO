"""Explore the relationship between vaso peak response and BOLD amplitude"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background')

zscores = pd.read_csv('results/zScoreData.csv')


peakLayerList = []
amplitudeList = []
subList = []

for sub in ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']:

    tmpVASO = zscores.loc[(zscores['runType'] == 'blockStim')
                      & (zscores['focus'] == 'v1')
                      & (zscores['contrast'] == 'visuotactile')
                      & (zscores['runType'] != 'blockStimLongTR')
                      & (zscores['subject'] == sub)
                      & (zscores['modality'] == 'VASO')
                      & (zscores['statType'] == 'zstat')
    ]

    peakLocVASO = tmpVASO.loc[tmpVASO['data'] == np.amax(tmpVASO['data']), 'layer'].iloc[0]

    tmpBOLD = zscores.loc[(zscores['runType'] == 'blockStim')
                          & (zscores['focus'] == 'v1')
                          & (zscores['contrast'] == 'visuotactile')
                          & (zscores['runType'] != 'blockStimLongTR')
                          & (zscores['subject'] == sub)
                          & (zscores['modality'] == 'BOLD')
                          & (zscores['statType'] == 'zstat')

                          ]

    amplitude = np.amax(tmpBOLD['data'])

    peakLayerList.append(peakLocVASO)
    amplitudeList.append(amplitude)
    subList.append(f'{sub}')


data = pd.DataFrame({'subject': subList, 'VASOpeakLayer':peakLayerList, 'BOLDamplitude': amplitudeList})


sns.scatterplot(data=data, x='VASOpeakLayer', y='BOLDamplitude', hue='subject')
plt.ylabel('BOLD zscore')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.tight_layout()
plt.savefig(f'/Users/sebastiandresbach/Desktop/peakVsAmpl.png', bbox_inches='tight')
plt.show()


