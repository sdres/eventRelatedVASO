import os
import glob
import nibabel as nb
import numpy as np
import subprocess
import pandas as pd
import json


subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']

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
                eventResults[focus][sub][base] = {}


                for modality in modalities:
                    eventResults[focus][sub][base][modality] = {}

                    print(f'processing {base}')

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



np.save('../my_dict.npy',  eventResults)
my_dict_back = np.load('../my_dict.npy',allow_pickle=True)



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
                            for key, value in eventResults[focus][sub][base][modality][trialType][f'layer {j}'].items():
                                subTrials.append(key)

                            for trial in subTrials[:-1]:

                                for n in range(len(eventResults[focus][sub][base][modality][trialType][f'layer {j}'][trial][0])):

                                    if modality == "BOLD":
                                        dataList.append(eventResults[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                    if modality == "VASO":
                                        dataList.append(-eventResults[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

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
                        for key, value in eventResults[focus][sub][base][modality][trialType][f'layer {j}'].items():
                            subTrials.append(key)

                        for trial in subTrials[:-1]:

                            for n in range(len(eventResults[focus][sub][base][modality][trialType][f'layer {j}'][trial][0])):

                                if modality == "BOLD":
                                    dataList.append(eventResults[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

                                if modality == "VASO":
                                    dataList.append(-eventResults[focus][sub][base][modality][trialType][f'layer {j}'][trial][0][n])

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
#         plt.title(f"{sub}  Event-Related Average", fontsize=20)
#         plt.legend(loc='upper right')

        ticks = range(0,14)
        labels = (np.arange(-4,10)*1.3).round(decimals=1)

        plt.xticks(ticks,labels,rotation=45)


        sns.lineplot(data=layerEventData, x='x', y='data', hue='modality', alpha=0.3, ci=None,palette=palette)
        # sns.lineplot(data=eventData.loc[(eventData['focus']=="v1")&(eventData['subject']=='sub-05')&(eventData['run'].str.contains('run-001'))], x='x', y='data', hue='modality', alpha=0.3, ci=None)
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
