"""Extract zscores across cortical depth to compare long and short TR acquisitions"""

import nibabel as nb
import glob
import os
import pandas as pd

# Set folders
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
DERIVATIVES = f'{ROOT}/derivativesTestTest'

# Set participants where we compared short and long TRs
subs = ['sub-09', 'sub-11']

modalities = ['BOLD', 'VASO']

# Initiate lists
subList = []
modalityList = []
layerList = []
runList = []
focusList = []
dataList = []
lengthList = []

for sub in subs:
    print(sub)

    # Find session in which long TRs were tested
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-00*/func/{sub}_ses-00*_task-*LongTR*_cbv.nii.gz'))

    sessions = []

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        # Find sessions with long vs short TR comparison
        if 'ses-001' in base:
            sessions.append('ses-001')
        if 'ses-002' in base:
            sessions.append('ses-002')
    sessions = set(sessions)  # Get rid of duplicates

    for ses in sessions:
        for runType in ['blockStim', 'blockStimLongTR']:
            for modality in modalities:
                # Find statistical map of desired runtype and modality
                statMap = f'{DERIVATIVES}/{sub}/{ses}/func/upsample/{sub}_{ses}_task-{runType}_contrast-visuotactile_{modality}_model-conv_zstat_ups5x.nii'
                nii = nb.load(statMap)
                data = nii.get_fdata()

                for focus in ['v1', 's1']:
                    try:
                        maskFile = f'{DERIVATIVES}/{sub}/{ses}/func/{sub}_masks/{sub}_{focus}_11layers_layers_equidist.nii'
                        maskNii = nb.load(maskFile)
                        maskData = maskNii.get_fdata()
                        # idxMask = maskData > 0
                    except:
                        print(f'no mask found for {focus}')
                        continue

                    # =============================================================================================
                    # Extract scores
                    # Mask data

                    for j in range(1, 12):  # Compute bin averages
                        layerRoi = maskData == j
                        mask_mean = np.mean(data[layerRoi.astype(bool)])

                        subList.append(sub)
                        dataList.append(mask_mean)
                        modalityList.append(modality)
                        focusList.append(focus)
                        layerList.append(j)
                        if 'LongTR' in runType:
                            lengthList.append('long TR')
                        else:
                            lengthList.append('short TR')

# Store data in DataFrame
zscoreData = pd.DataFrame({'subject': subList,
                         'modality': modalityList,
                         'data': dataList,
                         'layer': layerList,
                         'focus': focusList,
                         'TRlength': lengthList
                         }
                        )

# Save to .csv file
zscoreData.to_csv('results/zscoreLongVsShort.csv', sep=',', index=False)
