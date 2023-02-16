'''

Here we want to approximate the aread under the curve of zscores for block- and
event-wise stimulation. This is done separately for BOLD and VASO.

'''
import pandas as pd
import numpy as np



# load data
zscores = pd.read_csv('../results/backupOld/blocksVsEventsData.csv')

results = {}

for modality in ['VASO', 'BOLD']:
    results[modality] = {}

    for stimType in ['eventStim','blockStim']:
        results[modality][stimType] = {}
        # Limit plotting to modality, focus and visiotactile stimulation and remove
        # long TR block stimulation for fair comparisons
        tmp = zscores.loc[
            (zscores['modality']==modality)
            & (zscores['focus']=='v1')
            & (zscores['contrast']=='visiotactile')
            & (zscores['stimType']!='blockStimLongTR')
            & (zscores['runType']==stimType)
            ]


        tmpAUC = 0

        for layer in zscores['layer'].unique():

            tmpData = np.mean(tmp['data'].loc[tmp['layer']==layer])
            tmpAUC = tmpAUC + tmpData

        results[modality][stimType] = tmpAUC


# calculate detection sensitivity loss
for modality in ['VASO', 'BOLD']:
    loss = (
        (
        results[modality]['blockStim']
        - results[modality]['eventStim']
        )
        / results[modality]['blockStim']
        )

    print(f'When going from block- to event-wise stimulation, '
        f'the loss is {loss} for {modality}')
