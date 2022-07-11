import numpy as np
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import os
from matplotlib.ticker import FormatStrFormatter

subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']

root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'


############################
########### FIR ############
############################


layers = {'1':'deep','2':'middle','3':'superficial'}

subList = []
runList = []
modalityList = []
layerList = []
timepointList = []
dataList = []
focusList = []

for sub in ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-12','sub-13', 'sub-14']:
# for sub in ['sub-14']:
    runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-event*_run-00*_cbv.nii.gz'))
    for focus in ['v1', 's1']:
        for run in runs:

            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(base)
            if 'ses-001' in base:
                ses = 'ses-001'
            else:
                ses= 'ses-002'

            if sub == 'sub-09' and '4' in run:
                print('skipping')
                continue
            for modality in ['BOLD', 'VASO']:

                for timepoint in range(1,11):
                    try:
                        dataFile = f'{root}/derivatives/{sub}/{ses}/{base}_{focus}_{modality}_layers_FIR.feat/stats/pe{timepoint}.nii.gz'
                        dataNii = nb.load(dataFile)
                        data = dataNii.get_fdata()
                    except:
                        print(f'data missing')
                        continue


                    for layer in layers.keys():
                        subList.append(sub)
                        runList.append(base)
                        modalityList.append(modality)
                        layerList.append(layers[layer])
                        timepointList.append(timepoint)
                        focusList.append(focus)
                        if modality =='BOLD':
                            dataList.append(data[int(layer)-1])
                        if modality =='VASO':
                            dataList.append(-data[int(layer)-1])


FIRdata = pd.DataFrame({'subject':subList, 'run':runList, 'layer':layerList, 'modality':modalityList, 'data':dataList, 'volume':timepointList, 'focus':focusList})





layers = {'1':'deep','2':'middle','3':'superficial'}
eventResults2 = np.load('../data/trialWiseResponses.npy',allow_pickle=True).item()


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

plt.style.use('dark_background')

v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}




for focus in ['v1']:
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(7.5,10))

    sns.lineplot(ax=ax1, data=FIRdata.loc[FIRdata['focus']==focus], x="volume", y="data", hue='modality',palette=cmap,linewidth=2)

    ax1.set_ylabel(r'signal change [%]', fontsize=24)
    yLimits = ax1.get_ylim()

    ax1.set_ylim(0,yLimits[1])
    ax1.set_yticks(range(-1,int(yLimits[1])+1),fontsize=18)

    # prepare x-ticks
    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)
    for k,label in enumerate(labels):
        if (label - int(label) == 0):
            labels[k] = int(label)

    ax1.yaxis.set_tick_params(labelsize=18)
    ax1.xaxis.set_tick_params(labelsize=18)

    # tweak x-axis
    ax1.set_xticks(ticks[::2])
    ax1.set_xticklabels(labels[::2],fontsize=18)
    ax1.set_xlabel('Time [s]', fontsize=24)

    # draw lines
    ax1.axvspan(0, 2/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
    ax1.axhline(0,linestyle='--',color='white')




    # Set up second plot
    data = efficiencyData.loc[(efficiencyData['focus']==focus)]

    sns.lineplot(ax= ax2, data=data, x='nrTrials', y='score', hue='modality', palette= v1Palette, linewidth=2)


    ax2.set_ylabel(f"error", fontsize=24)
    ax2.set_xlabel('averaged trials', fontsize=24)


    ax2.yaxis.set_tick_params(labelsize=18)
    ax2.xaxis.set_tick_params(labelsize=18)
    ax2.set_ylim(0,25)


    ax2.vlines(timePoint, ymin=0,ymax=30, label='5% error', linestyle='dashed',color='white')



    ax2.legend().remove()
    ax1.legend().remove()
    fig.tight_layout()

    plt.savefig(f'../results/group_{focus}_eventResults.png', bbox_inches = "tight")
    plt.show()



#####################################################
################ Single subject plot ################
#####################################################

subList = []
dataList = []
modalityList = []
layerList = []
stimTypeList = []
focusList = []
contrastList = []
runList = []

root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'



for sub in ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']:
# for sub in ['sub-14']:
    # allRuns = sorted(glob.glob(f'{root}/{sub}/*/func/{sub}_*_task-*run-00*_cbv.nii.gz'))
    print(sub)

    blockRuns = sorted(glob.glob(f'{root}/{sub}/*/func/{sub}_*_task-block*run-00*_cbv.nii.gz'))

    # see whether there are multiple sessions
    sessions = []
    for run in blockRuns:
        if 'ses-001' in run:
            sessions.append('ses-001')
        if 'ses-002' in run:
            sessions.append('ses-002')
    sessions = set(sessions)

    for ses in sessions:
        print(f'{ses}')
        focuses = []
        for focus in ['v1', 's1']:
            if not sub == 'sub-07':
                try:
                    mask = nb.load(f'{root}/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_11layers_layers_equidist.nii').get_fdata()
                    focuses.append(focus)
                except:
                    print(f'{focus} not found')

            if sub == 'sub-07':
                try:
                    mask = nb.load(f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_11layers_eventStim_layers_equidist.nii').get_fdata()
                    focuses.append(focus)
                except:
                    print(f'{focus} not found')
        print(f'found ROIs for: {focuses}')


        for focus in focuses:
            if not sub == 'sub-07':
                mask = nb.load(f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_11layers_layers_equidist.nii').get_fdata()

            if sub == 'sub-07':
                mask = nb.load(f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_11layers_eventStim_layers_equidist.nii').get_fdata()

            for task in ['eventStimRandom', 'eventStim', 'eventStimVisOnly']:

                eventRuns = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_*_task-{task}_run-00*_cbv.nii.gz'))

                if len(eventRuns)==0:
                    print(f'no runs found for {task}.')
                    continue

                for modality in ['BOLD','VASO']:


                    if task == 'eventStim':
                        data = nb.load(f'{root}/derivatives/{sub}/{ses}/{sub}_{ses}_task-{task}_secondLevel_{modality}.gfeat/cope1.feat/stats/zstat1_scaled.nii.gz').get_fdata()

                        for j in range(1,12):  # Compute bin averages
                            layerRoi = mask == j
                            mask_mean = np.mean(data[layerRoi.astype(bool)])

                            subList.append(sub)
                            dataList.append(mask_mean)
                            modalityList.append(modality[:4])
                            layerList.append(j)
                            stimTypeList.append(task)
                            contrastList.append('visiotactile')
                            focusList.append(focus)
                            runList.append('eventStim')

                    elif task == 'eventStimRandom':
                        for i, contrast in enumerate(['visual', 'visiotactile', 'visual&visiotactile', 'visiotactile > visual'], start = 1):
                            data = nb.load(f'{root}/derivatives/{sub}/{ses}/{sub}_{ses}_task-{task}_secondLevel_{modality}.gfeat/cope{i}.feat/stats/zstat1_scaled.nii.gz').get_fdata()

                            for j in range(1,12):  # Compute bin averages
                                layerRoi = mask == j
                                mask_mean = np.mean(data[layerRoi.astype(bool)])

                                subList.append(sub)
                                dataList.append(mask_mean)
                                modalityList.append(modality[:4])
                                layerList.append(j)
                                stimTypeList.append(task)
                                contrastList.append(contrast)
                                focusList.append(focus)
                                runList.append('eventStim')



zscores = pd.DataFrame({'subject': subList, 'data': dataList, 'modality': modalityList, 'layer':layerList, 'stimType':stimTypeList, 'contrast':contrastList,'focus':focusList, 'runType':runList})


plt.style.use('dark_background')

v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}




for focus in ['v1']:
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(7.5,10))

    sns.lineplot(ax=ax1, data=FIRdata.loc[FIRdata['focus']==focus], x="volume", y="data", hue='modality',palette=v1Palette,linewidth=2)

    ax1.set_ylabel(r'Signal [$\beta$]', fontsize=24)
    yLimits = ax1.get_ylim()

    ax1.set_ylim(0,yLimits[1])
    ax1.set_yticks(range(-1,int(yLimits[1])+1),fontsize=18)

    # prepare x-ticks
    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)
    for k,label in enumerate(labels):
        if (label - int(label) == 0):
            labels[k] = int(label)

    ax1.yaxis.set_tick_params(labelsize=18)
    ax1.xaxis.set_tick_params(labelsize=18)

    # tweak x-axis
    ax1.set_xticks(ticks[::2])
    ax1.set_xticklabels(labels[::2],fontsize=18)
    ax1.set_xlabel('Time [s]', fontsize=24)

    # draw lines
    ax1.axvspan(0, 2/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
    ax1.axhline(0,linestyle='--',color='white')




    # Set up second plot
    data = zscores.loc[(zscores['contrast']=='visual&visiotactile')]

    sns.lineplot(ax= ax2, data=data, x='layer', y='data', hue='modality', palette= v1Palette, linewidth=2)

    ax2.set_xlabel('WM                                                CSF', fontsize=24)
    ax2.set_xticks([])

    ax2.set_ylabel(f'Z-score', fontsize=24)

    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))

    ax2.yaxis.set_tick_params(labelsize=18)


    ax2.legend().remove()
    ax1.legend().remove()
    fig.tight_layout()

    plt.savefig(f'../results/sub-14_{focus}_eventResults.png', bbox_inches = "tight")
    plt.show()




import matplotlib.gridspec as gridspec

palettesLayers = {'VASO':['#1f77b4','#7dadd9','#c9e5ff'],
'BOLD':['#ff7f0e', '#ffae6f','#ffdbc2']}


fig = plt.figure(tight_layout=True,figsize=(7.5,10))
gs = gridspec.GridSpec(2, 2)


ax3 = fig.add_subplot(gs[1, :])

for i, modality in enumerate(['BOLD', 'VASO']):

    data = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['focus']=='v1')]
    ax = fig.add_subplot(gs[0, i])

    sns.lineplot(ax=ax, data=data , x="volume", y="data", hue='layer',palette=palettesLayers[modality],linewidth=2)


    yLimits = ax.get_ylim()
    ax.set_ylim(-2,9)
    ax.set_yticks(range(-2,10,2),fontsize=18)

    # prepare x-ticks
    ticks = range(1,12,2)
    labels = (np.arange(0,11,2)*1.3).round(decimals=1)
    for k,label in enumerate(labels):
        if (label - int(label) == 0):
            labels[k] = int(label)

    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)

    if i == 0:
        ax.set_ylabel(r'Signal [$\beta$]', fontsize=24)
    else:
        ax.set_ylabel(r'', fontsize=24)
        ax.set_yticks([])

    ax.legend(loc='upper right',fontsize=12)

    # tweak x-axis
    ax.set_xticks(ticks[::2])
    ax.set_xticklabels(labels[::2],fontsize=18)
    ax.set_xlabel('Time [s]', fontsize=24)
    ax.set_title(modality, fontsize=24)

    # draw lines
    ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
    ax.axhline(0,linestyle='--',color='white')

# Set up second plot
data = zscores.loc[(zscores['contrast']=='visiotactile')&(zscores['focus']=='v1')]

sns.lineplot(ax= ax3, data=data, x='layer', y='data', hue='modality', palette= v1Palette, linewidth=2)

ax3.set_xlabel('WM                                                CSF', fontsize=24)
ax3.set_xticks([])
ax3.legend().set_title('')
ax3.set_ylabel(f'Z-score', fontsize=24)
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax3.yaxis.set_tick_params(labelsize=18)
ax3.legend(loc='upper left',fontsize=12)

plt.savefig(f'../results/groupLevel_{focus}_eventResults_withLayers.png', bbox_inches = "tight")


plt.show()




fig = plt.figure(tight_layout=True,figsize=(10.394,6.299))

for i, modality in enumerate(['BOLD', 'VASO']):

    data = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['focus']=='v1')]
    ax = fig.add_subplot(gs[0, i])

    sns.lineplot(ax=ax, data=data , x="volume", y="data", hue='layer',palette=palettesLayers[modality],linewidth=2)


    yLimits = ax.get_ylim()
    ax.set_ylim(-2,9)
    ax.set_yticks(range(-2,10,2),fontsize=18)

    # prepare x-ticks
    ticks = range(1,12,2)
    labels = (np.arange(0,11,2)*1.3).round(decimals=1)
    for k,label in enumerate(labels):
        if (label - int(label) == 0):
            labels[k] = int(label)

    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)

    if i == 0:
        ax.set_ylabel(r'Signal [$\beta$]', fontsize=24)
    else:
        ax.set_ylabel(r'', fontsize=24)
        ax.set_yticks([])

    ax.legend(loc='upper right',fontsize=12)

    # tweak x-axis
    ax.set_xticks(ticks[::2])
    ax.set_xticklabels(labels[::2],fontsize=18)
    ax.set_xlabel('Time [s]', fontsize=24)
    ax.set_title(modality, fontsize=24)

    # draw lines
    ax.axvspan(1, 3/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
    ax.axhline(0,linestyle='--',color='white')

plt.savefig(f'../results/groupLevel_{focus}_eventResults_withLayers.png', bbox_inches = "tight")


plt.show()
