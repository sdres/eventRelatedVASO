"""Extracting data from FIR"""

import numpy as np
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.gridspec as gridspec

# Disable pandas warining
pd.options.mode.chained_assignment = None

# Set folder where data is found
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

# Set participants to work on
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']

# Set modalities to work on
modalities = ['BOLD', 'VASO']
# modalities = ['BOLD']

# Define layer names
layerNames = {1: 'deep', 2: 'middle', 3: 'superficial'}

# Initialize lists
subList = []  # For participants
modalityList = []  # For modalities
layerList = []  # For layers
timepointList = []  # For volumes
dataList = []  # For extracted values
focusList = []  # For v1/s1 focus
contrastList = []

for sub in subs:  # Loop over participants

    # =========================================================================
    # Find session with event related runs
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/*/func/{sub}_*_task-event*run-00*_cbv.nii.gz'))

    # Initialte list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1,3):  # We had a maximum of 2 sessions
            if f'ses-00{i}' in run:
                sessions.append(f'ses-00{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    for ses in sessions:  # Loop over sessions
        # Set folder where layer-ROIs are stored
        sesFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'
        roiFolder = f'{sesFolder}/{sub}_masks'

        for focus in ['v1', 's1']:

            # Set layer file name
            if sub != 'sub-07':
                layerFile = f'{roiFolder}/{sub}_{focus}_3layers_layers_equidist.nii'
            if sub == 'sub-07':
                layerFile = f'{roiFolder}/{sub}_{focus}_3layers_eventStim_layers_equidist.nii'

            # Load layer file in nibabel
            layerNii = nb.load(layerFile)
            layerData = layerNii.get_fdata()  # Load as array
            layers = np.unique(layerData)[1:]  # Look for number of layers

            for modality in modalities:
                print(modality)

                for contrast in ['visual', 'visuotactile']:

                    maps = sorted(glob.glob(f'{sesFolder}/upsample/{sub}_{ses}_task-eventStim*-{contrast}_modality-{modality}_model-fir_*_ups5x.nii'))
                    print(f'found {len(maps)} maps')

                    for i, map in enumerate(maps):
                        try:
                            dataNii = nb.load(map)
                            data = dataNii.get_fdata()
                        except:
                            print(f'data missing')
                            continue

                        for layer in layers:

                            idxLayer = layerData == layer
                            mean = np.mean(data[idxLayer])

                            subList.append(sub)
                            modalityList.append(modality)
                            layerList.append(layerNames[layer])
                            timepointList.append(i)
                            focusList.append(focus)
                            contrastList.append(contrast)
                            dataList.append(mean)

FIRdata = pd.DataFrame({'subject': subList,
                        'layer': layerList,
                        'modality': modalityList,
                        'data': dataList,
                        'volume': timepointList,
                        'focus': focusList,
                        'contrast': contrastList
                        }
                       )
# Save to .csv file
zscores.to_csv('results/firData.csv', sep=',', index=False)

# =============================================================================
# Plotting
# ============================================================================

palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}

plt.style.use('dark_background')

gs = gridspec.GridSpec(2, 2)
fig = plt.figure(tight_layout=True,figsize=(10.394,6.299))

for i, modality in enumerate(['BOLD', 'VASO']):

    ax = fig.add_subplot(gs[0, i])

    for j, layer in enumerate(FIRdata['layer'].unique()):

        tmp = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['focus']=='v1')&(FIRdata['layer']==layer)&(FIRdata['contrast']=='visuotactile')]
        # tmp = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['focus']=='v1')&(FIRdata['layer']==layer)]

        # # Normalize to first timpoint
        # val = np.mean(tmp.loc[tmp['volume'] == 0]['data'])
        # tmp['data'] -= val

        # Plot layer data
        sns.lineplot(ax=ax, data=tmp , x="volume", y="data", color=palettesLayers[modality][j],linewidth=2, label=layer)

    yLimits = ax.get_ylim()
    ax.set_yticks(range(-2,14,2),fontsize=18)

    # prepare x-ticks
    ticks = range(0,11,2)
    labels = (np.arange(0,11,2)*1.3).round(decimals=1)
    for k,label in enumerate(labels):
        if (label - int(label) == 0):
            labels[k] = int(label)

    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)

    if i == 0:
        ax.set_ylabel(r'Signal [$\beta$]', fontsize=24)
    else:
        ax.set_ylabel(r'', fontsize=24)
        ax.set_yticks([])

    ax.legend(loc='upper right',fontsize=12)

    # tweak x-axis
    ax.set_xticks(ticks[::2])
    ax.set_xticklabels(labels[::2],fontsize=18)
    ax.set_xlabel('Time [s]', fontsize=24)
    ax.set_title(modality, fontsize=24)
    # draw lines
    ax.axvspan(0, 2/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
    ax.axhline(0,linestyle='--',color='white')

plt.savefig(f'/Users/sebastiandresbach/Desktop/sub-all_visuotactileOnly_{focus}_eventResults_withLayers.png', bbox_inches = "tight")


plt.show()
