'''

Populates .tsv files in subject-specific BIDS directory

'''


import pandas as pd
import glob
import os

ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

subs = ['sub-05']

for sub in subs:
    print(sub)


    runs = sorted(glob.glob(f'{ROOT}/{sub}/*/func/{sub}_*_task-*_run-00*_cbv.nii.gz'))

    sessions = []

    for run in runs:
        for sesNr in range(1,4):
            currSes = f'ses-00{sesNr}'
            if currSes in run:
                sessions.append(currSes)

    sessions = set(sessions)  # Remove duplicates

    for ses in sessions:
        funcDir = f'{ROOT}/derivatives/{sub}/{ses}'

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]

            # events = pd.DataFrame(columns = ['onset', 'duration', 'trial_type'])


            file = f'{funcDir}/events/{base}.txt'
            tmp = pd.read_csv(file, sep=' ', names = ['onset', 'duration', 'trial_type'])
            tmp['trial_type'] = ['stimulation']*len(tmp['trial_type'])

            tmp.to_csv(f'{ROOT}/{sub}/{ses}/func/{base}_events.tsv', sep = ' ',index=False)
