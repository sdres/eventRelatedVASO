import numpy as np
import glob
import pandas as pd
import os
import re

# define root dir
root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'
# define subjects to work on
subs = ['sub-05', 'sub-06', 'sub-07']


for sub in subs:
    # get all runs of all sessions
    runs = sorted(glob.glob(f'/{root}/{sub}/ses-*/func/{sub}_ses-*_task-*_run-00*_cbv.nii.gz'))

    for run in runs:
        # get basename of current run
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        # see session in which it was acquired
        if 'ses-001' in base:
            ses='ses-001'
        if 'ses-002' in base:
            ses='ses-002'
        # Because runs with randomized stimulation (visual vs visiotactile) will
        # be treated later we will skip them here
        if 'Random' in base:
            continue
        # get logfile
        logFile = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/events/{base}.log'
                                ,usecols=[0])

        # Because the column definition will get hickups if empty colums are
        # present, we find line with first trigger to then load the file anew,
        # starting with that line
        for index, row in logFile.iterrows():
            if re.search('Keypress: 5', str(row)):
                firstVolRow = index
                break
        # define column names
        ColNames = ['startTime', 'type', 'event']
        # load logfile again, starting with first trigger
        logFile = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/events/{base}.log', sep = '\t',skiprows=firstVolRow, names = ColNames)

        # initiate lists
        stimStart = []
        stimStop = []
        # loop over lines and fine stimulation start and stop times
        for index, row in logFile.iterrows():
            if re.search('stimulation started', logFile['event'][index]):
                stimStart.append(logFile['startTime'][index])
            if re.search('stimulation stopped', logFile['event'][index]):
                stimStop.append(logFile['startTime'][index])

        # convert lists to arrays and compute stimulation durations
        durs = np.asarray(stimStop) - np.asarray(stimStart)

        # make dataframe and save as text file
        design = pd.DataFrame({'startTime': stimStart, 'duration': durs, 'mod' : np.ones(len(durs))})
        np.savetxt(f'{root}/derivatives/{sub}/{ses}/events/{base}.txt', design.values, fmt='%1.2f')

    #######################
    ### For random runs ###
    #######################

    runs = sorted(glob.glob(f'/{root}/{sub}/{ses}/func/{sub}_{ses}_task-*Random_run-00*_cbv.nii.gz'))
    for run in runs:

        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        logFile = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/events/{base}.log', usecols=[0])

        for index, row in logFile.iterrows():
            if re.search('Keypress: 5', str(row)):
                firstVolRow = index
                break

        ColNames = ['startTime', 'type', 'event']
        logFile = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/events/{base}.log', sep = '\t',skiprows=24, names = ColNames)

        for condition in ['visual', 'visiotactile']:

            stimStart = []
            stimStop = []

            for index, row in logFile.iterrows():
                if re.search(f'{condition} stimulation started', logFile['event'][index]):
                    stimStart.append(logFile['startTime'][index])
                if re.search('stimulation stopped', logFile['event'][index]):
                    # only log if a start time was detected for this condition
                    if not len(stimStart)==len(stimStop):
                        stimStop.append(logFile['startTime'][index])


            durs = np.asarray(stimStop) - np.asarray(stimStart)


            design = pd.DataFrame({'startTime': stimStart, 'duration': durs, 'mod' : np.ones(len(durs))})
            np.savetxt(f'{root}/derivatives/{sub}/{ses}/events/{base}_{condition}.txt', design.values, fmt='%1.2f')
