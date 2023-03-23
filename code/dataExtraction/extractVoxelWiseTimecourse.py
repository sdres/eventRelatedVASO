"""Extracting average block timecourses"""

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
# subs = ['sub-14']

# Set modalities to work on
modalities = ['BOLD', 'VASO']
# modalities = ['BOLD']
                    
blockResults = {}
for focus in ['v1','s1']:
# for focus in ['v1']:
    print(focus)

    blockResults[focus] = {}
    for sub in subs:
        print(sub)

        blockRuns = sorted(glob.glob(f'{ROOT}/{sub}/*/func/{sub}_*_task-blockStim*_run-00*_cbv.nii.gz'))

        sessions = []
        for run in blockRuns:
            if 'ses-001' in run:
                sessions.append('ses-001')
            if 'ses-002' in run:
                sessions.append('ses-002')
        sessions = set(sessions)
        print(f'{sub} has {len(sessions)} session(s).')

        for ses in sessions:
            print(ses)
            sesFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'

            if not sub == 'sub-07':
                try:
                    maskFile = glob.glob(f'{sesFolder}/{sub}_masks/{sub}_{focus}_rim_down.nii.gz')[0]
                    mask = nb.load(maskFile).get_fdata()
                except:
                    print(f'No mask found for {focus}')
                    continue
            if sub == 'sub-07':
                try:
                    maskFile = glob.glob(f'{sesFolder}/{sub}_masks/{sub}_{focus}_rim_blockStim_down.nii.gz')[0]
                    mask = nb.load(maskFile).get_fdata()
                except:
                    print(f'No mask found for {focus}')
                    continue

            blockResults[focus][sub] = {}

            runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-blockSti*_run-00*_cbv.nii.gz'))

            for run in runs:
                # exclude runs with long TR because of different number of timepoints
                if 'Long' in run:
                    continue
                if 'Random' in run:
                    continue

                base = os.path.basename(run).rsplit('.', 2)[0][:-4]
                print(f'processing {base}')
                blockResults[focus][sub][base] = {}

                for modality in modalities:
                    blockResults[focus][sub][base][modality] = {}

                    print(modality)

                    run = f'{sesFolder}/{base}_{modality}.nii.gz'

                    Nii = nb.load(run)
                    # As before, get the data as an array.
                    data = Nii.get_fdata()
                    # load the nifty-header to get some meta-data.
                    header = Nii.header

                    nrVols = data.shape[-1]

                    # Or the TR, which is the 4th position of get_zooms().
                    tr = header.get_zooms()[3]

                    # Get scan duration in s
                    runTime = data.shape[-1]*tr

                    events = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/{base}_events.tsv', sep=' ')

                    # ============================================================
                    # compute baseline with non-stimulated periods
                    # ============================================================

                    bl = []  # Initiate list for baseline measurements
                    restOnset = 0  # First baseline starts at volume 0

                    for i, row in events.iterrows():
                        # Get trial onset
                        trialOnset = round(row['onset'])
                        # Add data from beginning of rest period until start of trial to baseline list
                        bl.append(data[..., int(round(restOnset/tr)):int(round(trialOnset/tr))][mask.astype(bool)])
                        # Set next rest onset
                        restOnset = trialOnset + row['duration']

                    if int(round(restOnset/tr)) < nrVols:
                        bl.append(data[..., int(round(restOnset/tr)):int(round(nrVols))][mask.astype(bool)])

                    means = []
                    for arr in bl:
                        means.append(np.mean(arr))
                    mean = sum(means) / len(means)

                    for i, row in events.iterrows():

                        onset = round(row['onset']/tr)
                        # Do the same for the offset
                        offset = round(onset + row['duration']/tr)

                        # check whether trial is fully acquired
                        if offset > data.shape[3]:
                            break

                        blockResults[focus][sub][base][modality][f'trial {i}'] = np.mean((((data[..., int(onset-4):int(offset + 8)][mask.astype(bool)]) / mean) - 1) * 100, axis=0)


subList = []
dataList = []
xList = []
modalityList = []
trialList = []
runList = []
focusList = []

for focus in ['s1', 'v1']:
    print(focus)

    for sub in subs:
        print(sub)
        runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-00*/func/{sub}_ses-001_task-blockStim_run-00*_cbv.nii.gz'))

        for run in runs:

            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(base)

            try:
                for modality in modalities:

                    subTrials = []
                    for key, value in blockResults[focus][sub][base][modality].items():
                        subTrials.append(key)

                    for trial in subTrials:

                        for n in range(len(blockResults[focus][sub][base][modality][trial])):

                            if modality == "BOLD":
                                dataList.append(blockResults[focus][sub][base][modality][trial][n])

                            if modality == "VASO":
                                dataList.append(-blockResults[focus][sub][base][modality][trial][n])

                            modalityList.append(modality)
                            trialList.append(trial)
                            runList.append(base)
                            xList.append(n)
                            subList.append(sub)
                            focusList.append(focus)
            except:
                print('data not available')

blockData = pd.DataFrame({'subject': subList,
                          'x': xList,
                          'data': dataList,
                          'modality': modalityList,
                          'trial': trialList,
                          'run': runList,
                          'focus': focusList
                          }
                         )
# Save to .csv file
blockData.to_csv('results/blockData.csv', sep=',', index=False)
