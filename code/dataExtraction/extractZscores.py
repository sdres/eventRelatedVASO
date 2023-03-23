"""Extracting zscores across cortical depth"""

import pandas as pd
import numpy as np
import nibabel as nb
import glob
import os

subList = []
dataList = []
modalityList = []
layerList = []
runTypeList = []
focusList = []
contrastList = []
sesList = []
designList = []
statTypeList = []
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
derivatives = f'{ROOT}/derivativesTestTest'

SUBS = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
# SUBS = ['sub-14']
for sub in SUBS:
    print(f'Working on {sub}')

    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/*/func/{sub}_*_task-*run-00*_cbv.nii.gz'))

    # Initialte list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1, 3):  # We had a maximum of 2 sessions
            if f'ses-00{i}' in run:
                sessions.append(f'ses-00{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    # Loop over sessions
    for ses in sessions:
        print(f'Working on {ses}')

        sesFolder = f'{derivatives}/{sub}/{ses}/func'

        focuses = []
        for focus in ['v1', 's1']:
            if not sub == 'sub-07':
                try:
                    mask = nb.load(f'{sesFolder}/{sub}_masks/{sub}_{focus}_11layers_layers_equidist.nii')
                    focuses.append(focus)
                except:
                    print(f'{focus} not found')

            if sub == 'sub-07':
                try:
                    mask = nb.load(f'{sesFolder}/{sub}_masks/{sub}_{focus}_11layers_blockStim_layers_equidist.nii')
                    focuses.append(focus)
                except:
                    print(f'{focus} not found')
        print(f'Found ROIs for: {focuses}')

        for focus in focuses:
            for modality in ['BOLD', 'VASO']:
                for statType in ['zstat', 'cope']:
                    statMaps = sorted(glob.glob(f'{sesFolder}/upsample/{sub}*{modality}*conv*{statType}*.nii'))

                    for statMap in statMaps:
                        base = os.path.basename(statMap)
                        inputs = base.split('_')
                        contrast = inputs[3].split('-')[-1]

                        runType = inputs[2].split('-')[-1]
                        design = runType[:9]

                        if sub != 'sub-07':
                            maskFile = f'{sesFolder}/{sub}_masks/{sub}_{focus}_11layers_layers_equidist.nii'
                            mask = nb.load(maskFile).get_fdata()
                        if sub == 'sub-07':
                            maskFile = f'{sesFolder}/{sub}_masks/{sub}_{focus}_11layers_{runType}_layers_equidist.nii'
                            mask = nb.load(maskFile).get_fdata()

                        data = nb.load(statMap).get_fdata()

                        for j in range(1, 12):  # Compute bin averages
                            layerRoi = mask == j
                            mask_mean = np.mean(data[layerRoi.astype(bool)])

                            subList.append(sub)
                            dataList.append(mask_mean)
                            modalityList.append(modality)
                            layerList.append(j)
                            runTypeList.append(runType)
                            contrastList.append(contrast)
                            focusList.append(focus)
                            sesList.append(ses)
                            designList.append(design)
                            statTypeList.append(statType)

# Store data in DataFrame
zscores = pd.DataFrame({'subject': subList,
                        'data': dataList,
                        'modality': modalityList,
                        'layer': layerList,
                        'contrast': contrastList,
                        'focus': focusList,
                        'runType': runTypeList,
                        'session': sesList,
                        'design': designList,
                        'statType': statTypeList
                        }
                       )
# Save to .csv file
zscores.to_csv('results/zScoreData.csv', sep=',', index=False)
