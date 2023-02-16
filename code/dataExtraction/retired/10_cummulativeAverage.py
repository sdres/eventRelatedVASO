import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import json


subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']
# subs = ['sub-05']

root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

modalities = ['BOLD', 'VASO']

eventResults = {}
for focus in ['v1']:

    eventResults[focus] = {}
    for sub in subs:
        print(sub)

        eventResults[focus][sub] = {}


        runsAll = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-eventStim*_run-00*_cbv.nii.gz'))

        sessions = []
        for run in runsAll:

            if 'ses-001' in run:
                if not any('ses-001' in s for s in sessions):
                    sessions.append('ses-001')
            if 'ses-002' in run:
                if not any('ses-002' in s for s in sessions):
                    sessions.append('ses-002')


        for ses in sessions:
            outFolder = f'{root}/derivatives/{sub}/{ses}/upsampledFunctional'

            if sub == 'sub-09' and focus == 's1Focus' and ses=='ses-002':
                continue

            runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-eventSti*_run-00*_cbv.nii.gz'))

            for run in runs:
                if 'Long' in run:
                    continue

                base = os.path.basename(run).rsplit('.', 2)[0][:-4]
                print(f'processing {base}')

                eventResults[focus][sub][base] = {}


                for modality in modalities:
                    eventResults[focus][sub][base][modality] = {}

                    print(f'{modality}')

                    run = f'{outFolder}/{base}_{focus}_{modality}.nii.gz'

                    if sub == 'sub-07':
                        mask = nb.load(f'{outFolder}/{sub}_{focus}_3layers_layers_eventStim_equidist.nii.gz').get_fdata()
                    else:
                        mask = nb.load(f'{outFolder}/{sub}_{focus}_3layers_layers_equidist.nii.gz').get_fdata()


                    idx_layers = np.unique(mask.astype("int"))
                    idx_layers = idx_layers[1:]


                    Nii = nb.load(run)
                    # As before, get the data as an array.
                    data = Nii.get_fdata()[:,:,:,:-2]
                    # load the nifty-header to get some meta-data.
                    header = Nii.header

                    # As the number of volumes which is the 4th position of
                    # get_shape. This seems to be unused so I will comment it out to check.
                    # nr_volumes = int(header.get_data_shape()[3])

                    # Or the TR, which is the 4th position of get_zooms().
                    tr = header.get_zooms()[3]

                     # Get scan duration in s
                    runTime = data.shape[-1]*tr


                    # Load information on run-wise motion
                    FDs = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/motionParameters/{base}_FDs.csv')

                    if 'Random' in base:

                        for eventType in ['visual', 'visiotactile']:

                            events = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/events/{base}_{eventType}.txt', sep = ' ', names = ['start','duration','trialType'])
                            eventResults[focus][sub][base][modality][eventType] = {}

                            for j in idx_layers:  # Compute bin averages

                                eventResults[focus][sub][base][modality][eventType][f'layer {j}'] = {}

                                layerRoi = mask == j


                                for i, row in events.iterrows():

                                    onset = round(row['start']/tr)
                                    # Do the same for the offset
                                    offset = round(onset + row['duration']/tr)

                                    # check whether trial is fully there
                                    if offset > data.shape[3]:
                                        break

                                    # Because we want the % signal-change, we need the mean
                                    # of the voxel we are looking at. This is done with
                                    # some fancy matrix operations.
                                    mask_mean = np.mean(data[:, :, :][layerRoi])

                                    # truncate motion data to event
                                    tmp = FDs.loc[(FDs['volume']>=(onset-2)/2)&(FDs['volume']<=(offset+2)/2)]

                                    if not (tmp['FD']>=2).any():
                                        eventResults[focus][sub][base][modality][eventType][f'layer {j}'][f'trial {i}'] = np.mean((((data[:, :, :, int(onset-4):int(offset+ 8)][layerRoi]) / mask_mean)- 1) * 100,axis=0)

                    if not 'Random' in base:

                        eventType = 'visiotactile'

                        events = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/events/{base}.txt', sep = ' ', names = ['start','duration','trialType'])
                        eventResults[focus][sub][base][modality][eventType] = {}

                        for j in idx_layers:  # Compute bin averages

                            eventResults[focus][sub][base][modality][eventType][f'layer {j}'] = {}

                            layerRoi = mask == j


                            for i, row in events.iterrows():

                                onset = round(row['start']/tr)
                                # Do the same for the offset
                                offset = round(onset + row['duration']/tr)

                                # check whether trial is fully there
                                if offset > data.shape[3]:
                                    break

                                # Because we want the % signal-change, we need the mean
                                # of the voxel we are looking at. This is done with
                                # some fancy matrix operations.
                                mask_mean = np.mean(data[:, :, :][layerRoi])

                                # truncate motion data to event
                                tmp = FDs.loc[(FDs['volume']>=(onset-2)/2)&(FDs['volume']<=(offset+2)/2)]

                                if not (tmp['FD']>=2).any():
                                    eventResults[focus][sub][base][modality][eventType][f'layer {j}'][f'trial {i}'] = np.mean((((data[:, :, :, int(onset-4):int(offset+ 8)][layerRoi]) / mask_mean)- 1) * 100,axis=0)

                            print(f'included {i} trials')



np.save('../../../data/trialWiseResponses.npy', eventResults)

type(eventResults)

eventResults2 = np.load('../../../data/trialWiseResponses.npy', allow_pickle=True).item()
eventResults = eventResults2
type(eventResults2)

subList = []
dataList = []
xList = []
modalityList = []
trialList = []
runList = []
layerList = []
conditionList = []

layers = {'1':'deep','2':'middle','3':'superficial'}


for focus, cmap in zip(['v1'],['tab10']):
    for sub in subs:
        print(sub)

        runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-event*_run-00*_cbv.nii.gz'))

        for run in runs:

            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(base)


            for modality in modalities:
                print(modality)

                if 'Random' in base:
                    for trialType in ['visual', 'visiotactile']:


                        for j in range(1, 4):

                            subTrials = []
                            for key, value in eventResults2[focus][sub][base][modality][trialType][f'layer {j}'].items():
                                subTrials.append(key)

                            for trial in subTrials[:-1]:

                                for n in range(len(eventResults2[focus][sub][base][modality][trialType][f'layer {j}'][trial][0])):

                                    if modality == "BOLD":
                                        dataList.append(eventResults2[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                    if modality == "VASO":
                                        dataList.append(-eventResults2[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                    modalityList.append(modality)
                                    trialList.append(trial)
                                    runList.append(base)
                                    xList.append(n)
                                    subList.append(sub)
                                    layerList.append(layers[str(j)])
                                    conditionList.append(trialType)
                else:
                    trialType = 'visiotactile'


                    for j in range(1, 4):

                        subTrials = []
                        for key, value in eventResults2[focus][sub][base][modality][trialType][f'layer {j}'].items():
                            subTrials.append(key)

                        for trial in subTrials[:-1]:

                            for n in range(len(eventResults2[focus][sub][base][modality][trialType][f'layer {j}'][trial][0])):

                                if modality == "BOLD":
                                    dataList.append(eventResults2[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                if modality == "VASO":
                                    dataList.append(-eventResults2[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                modalityList.append(modality)
                                trialList.append(trial)
                                runList.append(base)
                                xList.append(n)
                                subList.append(sub)
                                layerList.append(layers[str(j)])
                                conditionList.append(trialType)


layerEventData = pd.DataFrame({'subject': subList,'x':xList, 'data': dataList, 'modality': modalityList, 'trial': trialList, 'run':runList, 'layer':layerList, 'condition':conditionList})

trials = layerEventData.loc[layerEventData['subject']=='sub-05']
trials = trials['trial'].unique()

from random import seed
from random import choice
import matplotlib.pyplot as plt
import seaborn as sns

import nibabel as nb
import numpy as np
import imageio
import scipy
from scipy import ndimage
import os


folder = '/home/sebastian/Desktop/test'


seed(5)
palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
}
for sub in subs[3:4]:
    print(sub)

    # make list of all trials (some where excluded ue to motion)
    subTrials = []
    for key, value in eventResults['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['BOLD']['visiotactile']['layer 1'].items():
        subTrials.append(key)

    # initiate list to dump trials that were already included
    includedTrials = []
    # choose 40 random trials
    for n in range(40):
        # make a figure for each number of trials
        fig = plt.figure()

        # choose a random trial
        selection = choice(subTrials)
        # remove that trial from the list of possible trials
        subTrials.remove(selection)
        # add trialname to list of trials that were already included
        includedTrials.append(selection)

        # get VASO and BOLD responses of first included trial
        tmpBOLD = eventResults['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['BOLD']['visiotactile']['layer 1'][includedTrials[0]]
        tmpVASO = eventResults['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['VASO']['visiotactile']['layer 1'][includedTrials[0]]

        for layer in range(2,4):
            tmpBOLD = np.vstack((tmpBOLD, eventResults['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['BOLD']['visiotactile'][f'layer {layer}'][includedTrials[0]]))
            tmpVASO = np.vstack((tmpVASO, eventResults['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['VASO']['visiotactile'][f'layer {layer}'][includedTrials[0]]))

        for trial in range(0,len(includedTrials)):
            for layer in range(2,4):
                tmpBOLD = np.vstack((tmpBOLD, eventResults['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['BOLD']['visiotactile'][f'layer {layer}'][includedTrials[trial]]))
                tmpVASO = np.vstack((tmpVASO, eventResults['v1'][sub][f'{sub}_ses-001_task-eventStim_run-001']['VASO']['visiotactile'][f'layer {layer}'][includedTrials[trial]]))

        tmpVASO = np.mean(tmpVASO, axis=0)
        tmpBOLD = np.mean(tmpBOLD, axis=0)

        plt.plot(tmpBOLD, label='_nolegend_', color='tab:orange')
        plt.plot(-tmpVASO, label='_nolegend_', color='tab:blue')

        plt.ylabel('% signal change', fontsize=24)
        plt.xlabel('Time (s)', fontsize=24)


        ticks = range(0,14)
        labels = (np.arange(-4,10)*1.3).round(decimals=1)

        plt.xticks(ticks,labels,rotation=45)


        sns.lineplot(data=layerEventData, x='x', y='data', hue='modality', alpha=0.3, ci=None,palette=palette)

        plt.axvspan(4, 4+(2/1.3), color='grey', alpha=0.2, lw=0, label = 'stimulation')

        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
#         plt.xlabel('TR')
        plt.ylim(-2,2.5)
        plt.title(f"Average of {n+1} trials", fontsize=24, pad=20)
        plt.legend(loc='upper left',fontsize=12)
        plt.rcParams['savefig.facecolor']='white'

        plt.savefig(f'{folder}/{sub}_eventRelatedAveragesOf{str(n+1).zfill(2)}Trials.png', bbox_inches='tight')
        plt.show()

# make gif image
import imageio
images = []
imgs = sorted(glob.glob(f'{folder}/{sub}_eventRelatedAveragesOf*'))

for file in imgs:
    images.append(imageio.imread(file))
imageio.mimsave('/home/sebastian/Desktop/test/movie.gif', images, duration=0.5)



modalityMeans = {}

for focus in ['v1']:
    print(focus)
    modalityMeans[focus] = {}

    for modality in ['BOLD', 'VASO']:
        print(modality)
        modalityMeans[focus][modality] = {}

        tmp = np.zeros(14)

        for sub in subs:
            print(sub)

            if focus=='s1Focus' and sub == 'sub-09':
                continue

            runs = [*eventResults[focus][sub]]

            for run in runs:

                if 'Random' in run:
                    continue


                trials = [*eventResults[focus][sub][run][modality]['visiotactile']['layer 1']]

                for trial in trials:
                    nrTimePoints = len(eventResults[focus][sub][run][modality]['visiotactile']['layer 1'][trial][0])
                    if nrTimePoints == 14:
                        tmp = np.vstack((tmp, eventResults[focus][sub][run][modality]['visiotactile']['layer 1'][trial]))
                        for layer in range(2,4):
                            tmp = np.vstack((tmp, eventResults[focus][sub][run][modality]['visiotactile'][f'layer {layer}'][trial]))

        tmp = np.delete(tmp, (0), axis=0)
        tmpMean = np.mean(tmp, axis=0)
        tmpDemean = tmpMean - np.mean(tmpMean)

        # VASO is a negative contrast, therefore we have to switch the sign!
        if modality == 'VASO':
            tmpDemean = -tmpDemean

        modalityMeans[focus][modality] = tmpDemean

# plot mean responses to see whether they make sense
for focus in ['v1']:
    plt.figure()
    for modality in ['BOLD', 'VASO']:
        plt.plot(modalityMeans[focus][modality], label=f'{modality} mean')
    plt.legend()
    plt.title(focus)
    plt.show()

subList = []
runList = []
nrTrialsList = []
scoresList = []
modalityList = []
curentAverageList = []
focusList = []

# Loop over sensory areas. We will only look at visual cortex here
for focus in ['v1']:
    print(focus)
    for modality in ['BOLD', 'VASO']:
        print(modality)
        for sub in subs:
            print(sub)

            if focus=='s1' or sub == 'sub-09':
                continue
            # get subject-runs
            runs = [*eventResults[focus][sub]]

            for run in runs:
                # because we have a different number of trials for randomized stimulation, we will skip these runs
                # if 'Random' in run:
                #     continue

                # get list of trials that were included
                trials = [*eventResults[focus][sub][run][modality]['visiotactile']['layer 1']]
                print(len(trials))
                # loop over trials to get cummulative average
                for idx,trial in enumerate(trials, start=1):
                    # see whether trial window was completely acquired
                    nrTimePoints = len(eventResults[focus][sub][run][modality]['visiotactile']['layer 1'][trial][0])
                    # if not, go to next trial
                    if nrTimePoints != 14:
                        print(f'trial was not fully acquired')
                        continue

                    # create empty array to stack other arrays with
                    tmp = np.zeros(len(eventResults[focus][sub][run][modality]['visiotactile']['layer 1'][trial][0][3:-1]))

                    includedTrials = trials[:idx]

                    # stack trials until now
                    for includedTrial in includedTrials:

                        trial = eventResults[focus][sub][run][modality]['visiotactile']['layer 1'][includedTrial][0][3:-1]
                        for layer in range(2,4):
                            trial = np.vstack((trial, eventResults['v1'][sub][run][modality]['visiotactile'][f'layer {layer}'][includedTrial][0][3:-1]))

                        trial = np.mean(trial, axis=0)

                        demeanTrial = trial - np.mean(trial)
                        demeanTrialTrunc = demeanTrial.copy()
                        tmp = np.vstack((tmp, demeanTrialTrunc))

                    tmp = np.delete(tmp, (0), axis=0)
                    tmp = np.mean(tmp, axis=0)
                    tmp = tmp - np.mean(tmp)

                    if modality=='VASO':
                        tmp = -tmp

                    sumOfSquares = np.sum(np.subtract(modalityMeans[focus][modality][3:-1],tmp)**2)

                    subList.append(sub)
                    runList.append(run)
                    nrTrialsList.append(idx)
                    scoresList.append(sumOfSquares)
                    modalityList.append(modality)
                    curentAverageList.append(tmp.copy())
                    focusList.append(focus)

efficiencyData = pd.DataFrame({'subject':subList, 'run':runList, 'nrTrials':nrTrialsList, 'score':scoresList, "modality":modalityList, 'currentAverage': curentAverageList, 'focus':focusList})


v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}

for focus in ['v1']:
    data = efficiencyData.loc[(efficiencyData['focus']==focus)]
    # data = efficiencyData.loc[(efficiencyData['focus']==focus)&(efficiencyData['subject']==sub)]

    sns.lineplot(data=data, x='nrTrials', y='score', hue='modality', palette= v1Palette)

    plt.title('eventStim Response Stabilization', fontsize=24, pad=20)
#     plt.ylabel(f"sum of squares", fontsize=24)
    plt.ylabel(f"error", fontsize=24)
    plt.xlabel('averaged trials', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.legend(loc='upper right', fontsize=20)
    plt.ylim(0,30)
    plt.savefig(f'../results/stabilizingPerEvent.png', bbox_inches = "tight")
    plt.show()

# find 5% difference point
for modality in ['BOLD','VASO']:
    tmp = efficiencyData.loc[(efficiencyData['focus']=='v1')&(efficiencyData['modality']==modality)]

    firstTP = tmp['nrTrials'].unique()[0]
    firstTPVal = np.mean(tmp.loc[tmp['nrTrials']==firstTP]['score'])
    lastTP = tmp['nrTrials'].unique()[-1]
    lastTPVal = np.mean(tmp.loc[tmp['nrTrials']==lastTP]['score'])


    maxDiff = firstTPVal-lastTPVal
    percent = maxDiff/100
    thr = (percent*5)+lastTPVal
    thrLib = (percent*10)+lastTPVal
    critLib = 0
    for timePoint in tmp['nrTrials'].unique():
        val = np.mean(tmp.loc[tmp['nrTrials']==timePoint]['score'])
        if val <= thrLib:
            if critLib == 0:
                critLib = timePoint
                print(f'liberal critereon reached after trial {timePoint}')


        if val <= thr:
            print(f'{modality} critereon reached after trial {timePoint}')
            break


for focus in ['v1']:
    data = efficiencyData.loc[(efficiencyData['focus']==focus)]
    # data = efficiencyData.loc[(efficiencyData['focus']==focus)&(efficiencyData['subject']==sub)]

    sns.lineplot(data=data, x='nrTrials', y='score', hue='modality', palette= v1Palette)

    plt.title('eventStim Response Stabilization', fontsize=24, pad=20)
#     plt.ylabel(f"sum of squares", fontsize=24)
    plt.ylabel(f"error", fontsize=24)
    plt.xlabel('averaged trials', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(0,30)
    plt.vlines(critLib, ymin=0,ymax=30, label='10% error', linestyle='dashed',color='red')

    plt.vlines(timePoint, ymin=0,ymax=30, label='5% error', linestyle='dashed',color='black')
    plt.legend(loc='upper right', fontsize=20)
    plt.savefig(f'../results/stabilizingPerEventWithCrit.png', bbox_inches = "tight")
    plt.show()


##############################################
### Doing the same for layers individually ###
##############################################

layers = {'1':'deep','2':'middle','3':'superficial'}
eventResults = np.load('../../../data/trialWiseResponses.npy', allow_pickle=True).item()


modalityMeans = {}

for focus in ['v1']:
    print(focus)
    modalityMeans[focus] = {}

    for layer in range(1,4):
        modalityMeans[focus][layers[str(layer)]] = {}

        for modality in ['BOLD', 'VASO']:
            print(modality)
            modalityMeans[focus][layers[str(layer)]][modality] = {}

            tmp = np.zeros(14)

            for sub in subs:
                print(sub)

                if focus=='s1Focus' and sub == 'sub-09':
                    continue

                runs = [*eventResults[focus][sub]]

                for run in runs:

                    if 'Random' in run:
                        continue


                    trials = [*eventResults[focus][sub][run][modality]['visiotactile']['layer 1']]

                    for trial in trials:
                        nrTimePoints = len(eventResults[focus][sub][run][modality]['visiotactile'][f'layer {layer}'][trial][0])
                        if nrTimePoints == 14:
                            tmp = np.vstack((tmp, eventResults[focus][sub][run][modality]['visiotactile'][f'layer {layer}'][trial]))


            tmp = np.delete(tmp, (0), axis=0)
            tmpMean = np.mean(tmp, axis=0)
            tmpDemean = tmpMean - np.mean(tmpMean)

            # VASO is a negative contrast, therefore we have to switch the sign!
            if modality == 'VASO':
                tmpDemean = -tmpDemean

            modalityMeans[focus][layers[str(layer)]][modality] = tmpDemean

# plot mean responses to see whether they make sense
for focus in ['v1']:
    for modality in ['BOLD', 'VASO']:
        plt.figure()
        for layer in range(1,4):
            plt.plot(modalityMeans[focus][layers[str(layer)]][modality], label=f'{modality} {layers[str(layer)]} mean')
        plt.legend()
        plt.title(focus)
        plt.show()



# subs = ['sub-05']

subList = []
runList = []
nrTrialsList = []
scoresList = []
modalityList = []
curentAverageList = []
focusList = []
layerList = []

# Loop over sensory areas. We will only look at visual cortex here
for focus in ['v1']:
    print(focus)
    for modality in ['BOLD', 'VASO']:
        print(modality)
        for sub in subs:
            print(sub)

            if focus=='s1' or sub == 'sub-09':
                continue

            # get subject-runs
            runs = [*eventResults[focus][sub]]

            for run in runs:
                # because we have a different number of trials for randomized stimulation, we will skip these runs
                # if 'Random' in run:
                #     continue

                # get list of trials that were included
                trials = [*eventResults[focus][sub][run][modality]['visiotactile']['layer 1']]

                print(len(trials))

                for layer in range(1,4):


                    # loop over trials to get cummulative average
                    for idx,trial in enumerate(trials, start=1):
                        # see whether trial window was completely acquired
                        nrTimePoints = len(eventResults[focus][sub][run][modality]['visiotactile'][f'layer {layer}'][trial][0])

                        # if not, go to next trial
                        if nrTimePoints != 14:
                            print(f'trial was not fully acquired')
                            continue


                        includedTrials = trials[:idx]

                        # create empty array to stack other arrays with
                        tmp = np.zeros(len(eventResults[focus][sub][run][modality]['visiotactile'][f'layer {layer}'][trial][0][3:-1]))

                        # stack trials until now
                        for includedTrial in includedTrials:

                            trial = eventResults[focus][sub][run][modality]['visiotactile'][f'layer {layer}'][includedTrial][0][3:-1]



                            # trial = np.mean(trial, axis=0)

                            demeanTrial = trial - np.mean(trial)
                            demeanTrialTrunc = demeanTrial.copy()
                            tmp = np.vstack((tmp, demeanTrialTrunc))

                        tmp = np.delete(tmp, (0), axis=0)
                        tmp = np.mean(tmp, axis=0)
                        tmp = tmp - np.mean(tmp)

                        if modality=='VASO':
                            tmp = -tmp

                        sumOfSquares = np.sum(np.subtract(modalityMeans[focus][layers[str(layer)]][modality][3:-1],tmp)**2)

                        subList.append(sub)
                        runList.append(run)
                        nrTrialsList.append(idx)
                        scoresList.append(sumOfSquares)
                        modalityList.append(modality)
                        curentAverageList.append(tmp.copy())
                        focusList.append(focus)
                        layerList.append(layers[str(layer)])

efficiencyData = pd.DataFrame({'subject':subList, 'run':runList, 'nrTrials':nrTrialsList, 'score':scoresList, "modality":modalityList, 'currentAverage': curentAverageList, 'focus':focusList, 'layer':layerList})



# v1Palette = {
#     'BOLD': 'tab:orange',
#     'VASO': 'tab:blue'}

for focus in ['v1']:
    for modality in modalities:
        data = efficiencyData.loc[(efficiencyData['focus']==focus)&(efficiencyData['modality']==modality)]
        # data = efficiencyData.loc[(efficiencyData['focus']==focus)&(efficiencyData['subject']==sub)]

        sns.lineplot(data=data, x='nrTrials', y='score', hue='layer', palette= 'rocket')

        plt.title(f'{modality}', fontsize=24, pad=20)
    #     plt.ylabel(f"sum of squares", fontsize=24)
        plt.ylabel(f"error", fontsize=24)
        plt.xlabel('averaged trials', fontsize=24)
        plt.xticks(fontsize=18)
        plt.yticks(fontsize=18)
        plt.legend(loc='upper right', fontsize=20)
        # plt.ylim(0,30)
        plt.savefig(f'../results/stabilizingPerEventAcrossLayers{modality}.png', bbox_inches = "tight")
        plt.show()
