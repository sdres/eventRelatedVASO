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

# define root dir
root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'
# define subjects to work on
subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']
# subs = ['sub-05']


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
        # because the block/event logfiles have different codes at the start/end
        # of stimulation, we have to code them differently.
            if 'block' in run:
                if re.search('stimulation started', logFile['event'][index]):
                    stimStart.append(logFile['startTime'][index])
                if re.search('stimulation stopped', logFile['event'][index]):
                    stimStop.append(logFile['startTime'][index])
            if 'event' in run:
                if re.search('visual stimulation started', logFile['event'][index]):
                    stimStart.append(logFile['startTime'][index])
                if re.search('visual stimulation stopped', logFile['event'][index]):
                    stimStop.append(logFile['startTime'][index])
        # convert lists to arrays and compute stimulation durations
        durs = np.asarray(stimStop) - np.asarray(stimStart)

        # make dataframe and save as text file
        design = pd.DataFrame({'startTime': stimStart, 'duration': durs, 'mod' : np.ones(len(durs))})
        np.savetxt(f'{root}/derivatives/{sub}/{ses}/events/{base}.txt', design.values, fmt='%1.2f')


        # make design files to compare short vs long ITIs
        if 'blockStim' in base:
            continue

        meanDur = np.mean(durs)
        # the first ITI is due the time between start of run and first stimulus
        ITIs = [19]
        for i, trial in enumerate(stimStart[:-1]):
            tmp = (stimStart[i+1]-trial)-meanDur
            ITIs.append(tmp)

        # get max ITI
        ITImax = np.amax(ITIs[1:]) # exclude first, long ITI
        # get min ITI
        ITImin = np.amin(ITIs)
        # get middle ITI
        middleITI = ((ITImax-ITImin)/2)+ITImin
        medianITI = np.median(ITIs)

        usedTrials = []

        for type in ['longITI', 'shortITI']:
            # print(type)
            splitStarts = []
            splitDurs = []

            for i, ITI in enumerate(ITIs):
                if type == 'longITI' and ITI > medianITI:
                    # print(ITI)
                    splitStarts.append(stimStart[i])
                    splitDurs.append(durs[i])

                    if i in usedTrials:
                        print('double assignment of trial')
                    usedTrials.append(i)

                if type == 'shortITI' and ITI < medianITI:
                    # print(ITI)
                    splitStarts.append(stimStart[i])
                    splitDurs.append(durs[i])
                    if i in usedTrials:
                        print('double assignment of trial')

                    usedTrials.append(i)

            design = pd.DataFrame({'startTime': splitStarts, 'duration': splitDurs, 'mod' : np.ones(len(splitDurs))})
            np.savetxt(f'{root}/derivatives/{sub}/{ses}/events/{base}_{type}.txt', design.values, fmt='%1.2f')

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


# We might also want to know the distribution of inter trial intervals in order
# to check whether they are evenly distributed

meanDur = np.mean(durs)
ITIs = []
for i, trial in enumerate(stimStart[:-1]):
    tmp = (stimStart[i+1]-trial)-meanDur
    ITIs.append(tmp)

data = pd.DataFrame({'ITIs':ITIs})

fig, ax = plt.subplots()
sns.histplot(data=data, x="ITIs", binwidth=0.5)
# sns.kdeplot(data=data, x="ITIs", bw_adjust=.25)
plt.title(f'Inter-Trial Intervals',fontsize=24)
plt.ylabel('# Trials',fontsize=24)
plt.yticks(fontsize=14)
plt.xticks(fontsize=14)
plt.xlabel('ITI (seconds)',fontsize=20)
plt.savefig(f'../results/ITIsCount.png',bbox_inches='tight')
plt.show()

# We might also want to run the deconvolution for long and short ITIs
# separately.

# get max ITI
ITImax = np.amax(ITIs)
ITImin = np.amin(ITIs)

middleITI = ITImax-ITImin
subs = ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-12','sub-13', 'sub-14']:

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
            if re.search('visual stimulation started', logFile['event'][index]):
                stimStart.append(logFile['startTime'][index])
            if re.search('visual stimulation stopped', logFile['event'][index]):
                stimStop.append(logFile['startTime'][index])

        # convert lists to arrays and compute stimulation durations
        durs = np.asarray(stimStop) - np.asarray(stimStart)


        # make dataframe and save as text file
        design = pd.DataFrame({'startTime': stimStart, 'duration': durs, 'mod' : np.ones(len(durs))})
        np.savetxt(f'{root}/derivatives/{sub}/{ses}/events/{base}.txt', design.values, fmt='%1.2f')

        if 'blockStim' in base:
            continue

        meanDur = np.mean(durs)
        # the first ITI is due the time between start of run and first stimulus
        ITIs = [19]
        for i, trial in enumerate(stimStart[:-1]):
            tmp = (stimStart[i+1]-trial)-meanDur
            ITIs.append(tmp)

        # get max ITI
        ITImax = np.amax(ITIs[1:]) # exclude first, long ITI
        # get min ITI
        ITImin = np.amin(ITIs)
        # get middle ITI
        middleITI = ((ITImax-ITImin)/2)+ITImin
        medianITI = np.median(ITIs)

        usedTrials = []

        for type in ['longITI', 'shortITI']:
            # print(type)
            splitStarts = []
            splitDurs = []

            for i, ITI in enumerate(ITIs):
                if type == 'longITI' and ITI > medianITI:
                    # print(ITI)
                    splitStarts.append(stimStart[i])
                    splitDurs.append(durs[i])

                    if i in usedTrials:
                        print('double assignment of trial')
                    usedTrials.append(i)

                if type == 'shortITI' and ITI < medianITI:
                    # print(ITI)
                    splitStarts.append(stimStart[i])
                    splitDurs.append(durs[i])
                    if i in usedTrials:
                        print('double assignment of trial')

                    usedTrials.append(i)

            design = pd.DataFrame({'startTime': splitStarts, 'duration': splitDurs, 'mod' : np.ones(len(splitDurs))})
            np.savetxt(f'{root}/derivatives/{sub}/{ses}/events/{base}_{type}.txt', design.values, fmt='%1.2f')



from scipy.stats import chisquare
chisquare(ITIs[1:])
