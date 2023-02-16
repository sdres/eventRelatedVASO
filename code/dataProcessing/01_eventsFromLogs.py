'''

Reads .log files and outputs event files in BIDS format.

'''

import numpy as np
import glob
import pandas as pd
import os
import re

import seaborn as sns
import matplotlib.pyplot as plt

# define ROOT dir
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
# define subjects to work on
SUBS = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']

for sub in SUBS:
    print(f'Working on {sub}')

    # =========================================================================
    # Look for sessions
    # Collectall runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/*/func/{sub}_*_task-*run-00*_cbv.nii.gz'))

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

    # Loop over sessions
    for ses in sessions:
        print(f'Working on {ses}')

        # Set folder for session outputs
        # outFolderFSL = f'{ROOT}/derivativesTest/{sub}/{ses}/events'
        outFolderBIDS = f'{ROOT}/{sub}/{ses}/func'

        # Look for individual runs within session (containing both nulled and notnulled images)
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

        runTypes = []

        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*_run-00*_cbv.nii.gz'))
        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-12].split('-')[-1]
            if base not in runTypes:
                runTypes.append(base)

        for runType in runTypes:

            runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-00*_cbv.nii.gz'))

            for run in runs:
                print(run)

                base = os.path.basename(run).rsplit('.', 2)[0][:-4]

                # Set logfile
                logFile = f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log'

                # Load file as dataframe
                logData = pd.read_csv(logFile ,usecols = [0])

                # Because the column definition will get hickups if empty colums are
                # present, we find line with first trigger to then load the file anew,
                # starting with that line
                for index, row in logData.iterrows():
                    if re.search('Keypress: 5', str(row)):
                        firstVolRow = index
                        break

                # Define column names
                ColNames = ['startTime', 'type', 'event']
                # Load logfile again, starting with first trigger
                logData = pd.read_csv(logFile,
                                      sep = '\t',
                                      skiprows = firstVolRow,
                                      names = ColNames
                                      )

                # ==================================================================
                # Get design in BIDS style

                stimStart = []
                stimStop = []
                stimType = []

                for index, row in logData.iterrows():
                    if re.search(f'stimulation started', logData['event'][index]):
                        stimType.append(logData['event'][index].split(' ')[0])
                        stimStart.append(logData['startTime'][index])

                    if re.search('stimulation stopped', logData['event'][index]):
                        stimStop.append(logData['startTime'][index])

                if 'eventStimVisOnly' in base:
                    stimStart = stimStart[::2]

                durs = np.asarray(stimStop) - np.asarray(stimStart)


                if not (('Random' in runType) and not ('VisOnly' in runType)):
                    stimType = ['visiotactile']*len(durs)

                if 'Random' in runType:
                    stimType = stimType

                if 'VisOnly' in runType:
                    stimType = ['visual']*len(durs)

                design = pd.DataFrame({'onset': stimStart,
                                       'duration': durs,
                                       'trial_type': stimType
                                       }
                                      )
                design.to_csv(f'{outFolderBIDS}/{base}_events.tsv',
                              sep = ' ',
                              index = False
                              )
