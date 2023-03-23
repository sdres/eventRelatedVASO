"""Extracting data from FIR"""

import numpy as np
import glob
import nibabel as nb
import pandas as pd
import os

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
layerNames = {1: 'Deep', 2: 'Middle', 3: 'Superficial'}

# Initialize lists
subList = []  # For participants
modalityList = []  # For modalities
layerList = []  # For layers
timepointList = []  # For volumes
dataList = []  # For extracted values
focusList = []  # For v1/s1 focus
contrastList = []
runTypeList = []

for sub in subs:  # Loop over participants
    print(f'Processing {sub}')
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
            print(f'{focus}')
            try:
                # Set layer file name
                if sub != 'sub-07':
                    layerFile = f'{roiFolder}/{sub}_{focus}_3layers_layers_equidist.nii'
                if sub == 'sub-07':
                    layerFile = f'{roiFolder}/{sub}_{focus}_3layers_eventStim_layers_equidist.nii'

                # Load layer file in nibabel
                layerNii = nb.load(layerFile)
                layerData = layerNii.get_fdata()  # Load as array
                layers = np.unique(layerData)[1:]  # Look for number of layers
            except:
                print(f'No mask found for {focus}')
                continue

            for modality in modalities:
                print(modality)

                for contrast in ['visual', 'visuotactile']:

                    maps = sorted(glob.glob(f'{sesFolder}/upsample/{sub}_{ses}_task-eventStim*_contrast-{contrast}_modality-{modality}_model-fir_*_ups5x.nii'))
                    print(f'found {len(maps)} maps for {contrast}')

                    for i, map in enumerate(maps):
                        base = os.path.basename(map)
                        runType = base.split('_')[2][5:]
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
                            runTypeList.append(runType)

FIRdata = pd.DataFrame({'subject': subList,
                        'layer': layerList,
                        'modality': modalityList,
                        'data': dataList,
                        'volume': timepointList,
                        'focus': focusList,
                        'contrast': contrastList,
                        'runType': runTypeList
                        }
                       )
# Save to .csv file
FIRdata.to_csv('results/firData.csv', sep=',', index=False)