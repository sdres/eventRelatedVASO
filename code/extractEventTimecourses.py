import numpy as np
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import os

subs = ['sub-12']
ses = 'ses-001'
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

layerEventData

sns.lineplot(data=layerEventData, x="x", y="data", hue='modality')

############################
### plot individual runs ###
############################

# for focus, cmap in zip(['v1Focus'],['tab10']):
#     for stimulation in ['visual', 'visiotactile']:
sub = 'sub-12'
for run in layerEventData['run'].unique():
    g = sns.FacetGrid(layerEventData.loc[layerEventData['run']==run], col="modality", hue="layer", height= 5, aspect = 1.5,palette='rocket', sharey=False)
    g.map_dataframe(sns.lineplot, x="x", y="data")
    g.add_legend(bbox_to_anchor = (1, 1), borderaxespad = 0)

    #
    g.axes[0,0].set_ylabel('BOLD % signal change', fontsize=24)
    g.axes[0,1].set_ylabel('', fontsize=24)


    g.axes[0,0].set_title('BOLD', fontsize=24, pad=20)

    g.axes[0,1].set_title('VASO', fontsize=24, pad=20)

    plt.setp(g._legend.get_title(), fontsize=24)
    plt.setp(g._legend.get_texts(), fontsize=24)

    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)

    for ax in g.axes.reshape(-1):
        ax.axvspan(4, 4 + (2), color='grey', alpha=0.2, lw=0, label = 'stimulation on')
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels,fontsize=24)
        # ax.set_xticks(spacing)
        plt.setp(ax.get_yticklabels(), fontsize=24)
        # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax.set_xlabel('Time (s)', fontsize=24)

    # g.savefig(f"{root}/sub-09_layers.png", bbox_inces='tight')

    plt.show()






layers = {'1':'deep','2':'middle','3':'superficial'}

subList = []
runList = []
modalityList = []
layerList = []
timepointList = []
dataList = []
focusList = []
for sub in ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-12','sub-13', 'sub-14']:
# for sub in ['sub-13']:
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

                    # dataFile = f'{root}/derivatives/{sub}/ses-001/{base}_{modality}_vessel_FIR.feat/stats/pe{timepoint}.nii.gz'
                    # dataNii = nb.load(dataFile)
                    # data = dataNii.get_fdata()
                    #
                    # subList.append(sub)
                    # runList.append(base)
                    # modalityList.append(modality)
                    # layerList.append('vessel')
                    # timepointList.append(timepoint)
                    # if modality =='BOLD':
                    #     dataList.append(data[0])
                    # if modality =='VASO':
                    #     dataList.append(-data[0])

FIRdata = pd.DataFrame({'subject':subList, 'run':runList, 'layer':layerList, 'modality':modalityList, 'data':dataList, 'volume':timepointList, 'focus':focusList})


# for sub in ['sub-06', 'sub-07','sub-08', 'sub-09', 'sub-13', 'sub-14']:
for sub in ['sub-12']:
    fig, axes = plt.subplots(1,2)
    for i, modality in enumerate(['BOLD', 'VASO']):
        tmp = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['subject']==sub)]
        sns.lineplot(ax = axes[i], data=tmp, x="volume", y="data", hue='layer')
    plt.suptitle(sub, fontsize=20)
    plt.show()


for sub in ['sub-06', 'sub-07','sub-08', 'sub-09', 'sub-13', 'sub-14']:
    tmp1 = FIRdata.loc[FIRdata['subject']==sub]
    for run in tmp1['run'].unique():
        fig, axes = plt.subplots(1,2)
        for i, modality in enumerate(['BOLD', 'VASO']):
            tmp = tmp1.loc[(tmp1['modality']==modality)&(tmp1['run']==run)]
            sns.lineplot(ax = axes[i], data=tmp, x="volume", y="data", hue='layer')
        plt.suptitle(run, fontsize=20)
        plt.show()


v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}
s1Palette = {
    'BOLD': 'tab:green',
    'VASO': 'tab:blue'}

palettes = [v1Palette,s1Palette]

for focus, cmap in zip(['v1','s1'],palettes):


    fig, ax  = plt.subplots()
    sns.lineplot(data=FIRdata.loc[FIRdata['focus']==focus], x="volume", y="data", hue='modality',palette=cmap)
    plt.title('Group Finite Impulse Response', fontsize=24, pad=20)
    plt.ylabel(r'mean $\beta $', fontsize=24)
    yLimits = ax.get_ylim()
    plt.ylim(0,yLimits[1])
    plt.yticks(range(-1,int(yLimits[1])+1),fontsize=16)

    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)

    plt.xticks(ticks, labels, fontsize=16, rotation=45)
    plt.xlabel('Time (s)', fontsize=24)

    plt.axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')
    plt.legend(loc='upper right', fontsize=14)
    plt.savefig(f'{root}/Group_{focus}_FIR.png', bbox_inches = "tight")

    plt.show()




for sub in ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-12','sub-13', 'sub-14']:
    for focus, cmap in zip(['v1','s1'],palettes):


        fig, ax  = plt.subplots()
        sns.lineplot(data=FIRdata.loc[(FIRdata['focus']==focus)&(FIRdata['subject']==sub)], x="volume", y="data", hue='modality',palette=cmap)
        plt.title(f'{sub} Finite Impulse Response', fontsize=24, pad=20)
        plt.ylabel('% Signal Change', fontsize=24)
        yLimits = ax.get_ylim()
        plt.ylim(0,yLimits[1])
        plt.yticks(range(-1,int(yLimits[1])+1),fontsize=16)

        ticks = range(0,11)
        labels = (np.arange(0,11)*1.3).round(decimals=1)

        plt.xticks(ticks, labels, fontsize=16, rotation=45)
        plt.xlabel('Time (s)', fontsize=24)

        plt.axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')
        plt.legend(loc='upper right', fontsize=14)
        plt.savefig(f'{root}/{sub}_{focus}_FIR.png', bbox_inches = "tight")

        plt.show()



for sub in ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-12','sub-13', 'sub-14']:

    for focus, cmap in zip(['v1'],palettes):

        fig, axes = plt.subplots(1,2, figsize=(20,7))
        for i, modality in enumerate(['BOLD', 'VASO']):
            tmp = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['subject']==sub)]
            sns.lineplot(ax = axes[i], data=tmp, x="volume", y="data", hue='layer')
            axes[i].set_title(modality, fontsize=24)

        # plt.suptitle('Group Finite Impulse Response', fontsize=24)
        axes[0].set_ylabel('% Signal Change', fontsize=24)
        axes[1].set_ylabel('', fontsize=24)


        ticks = range(0,11)
        labels = (np.arange(0,11)*1.3).round(decimals=1)

        for i in range(2):
            yLimits = axes[i].get_ylim()

            # axes[i].set_yticklabels(range(-1,int(yLimits[1])+1),fontsize=20)
            axes[i].tick_params(axis='y', labelsize=20)
            axes[i].set_xticks(ticks, labels, fontsize=16, rotation=45)
            axes[i].set_xlabel('Time (s)', fontsize=24)
            axes[i].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')

        axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=14)
        axes[0].legend().remove()

        plt.savefig(f'{root}/{sub}_{focus}_FIR.png', bbox_inches = "tight")

        plt.show()





for focus, cmap in zip(['v1'],palettes):

    fig, axes = plt.subplots(1,2, figsize=(16,6))
    for i, modality in enumerate(['BOLD', 'VASO']):
        tmp = FIRdata.loc[(FIRdata['modality']==modality)]
        sns.lineplot(ax = axes[i], data=tmp, x="volume", y="data", hue='layer',palette='rocket')
        axes[i].set_title(modality, fontsize=32)

    # plt.suptitle('Group Finite Impulse Response', fontsize=24)
    axes[0].set_ylabel(r'mean $\beta $', fontsize=28)
    axes[1].set_ylabel('', fontsize=24)


    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)

    for i in range(2):
        # yLimits = axes[i].get_ylim()

        # ylabels = np.arange(int(yLimits[0])-1,int(yLimits[1]+2))

        # axes[i].set_yticklabels(ylabels)

        axes[i].tick_params(axis='y', labelsize=24)
        axes[i].set_xticks(ticks, labels, fontsize=24, rotation=45)
        axes[i].set_xlabel('Time (s)', fontsize=28)
        axes[i].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')

    axes[1].legend(bbox_to_anchor=(1.1, 1.05), fontsize=24)
    axes[0].legend().remove()

    plt.savefig(f'{root}/Group_{focus}_FIR.png', bbox_inches = "tight")

    plt.show()




from matplotlib.ticker import FormatStrFormatter

for focus, cmap in zip(['v1'],['Dark2']):

    g = sns.FacetGrid(FIRdata.loc[FIRdata['focus']==focus], col="modality", hue="layer", sharey=False, height= 5, aspect = 1.5,palette='rocket')
    g.map_dataframe(sns.lineplot, x="volume", y="data")
    g.add_legend(bbox_to_anchor = (1, 1), borderaxespad = 0)


    g.axes[0,0].set_ylabel(r'mean $\beta $', fontsize=24)

    g.axes[0,1].set_xlabel('TR', fontsize=24)
    g.axes[0,0].set_xlabel('TR', fontsize=24)


    g.axes[0,1].set_title('VASO', fontsize=24, pad=20)
    g.axes[0,0].set_title('BOLD', fontsize=24, pad=20)


    g.axes[0,0].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')
    g.axes[0,1].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')


    plt.setp(g._legend.get_title(), fontsize=24)
    plt.setp(g._legend.get_texts(), fontsize=24)



    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)

    g.axes[0,0].set_xticks(ticks)

    g.axes[0,0].set_xticklabels(labels,fontsize=24)
    g.axes[0,1].set_xticklabels(labels,fontsize=24)


    plt.setp(g.axes[0,0].get_yticklabels(), fontsize=24)
    plt.setp(g.axes[0,1].get_yticklabels(), fontsize=24)


    g.axes[0,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    g.axes[0,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

#     plt.legend(bbox_to_anchor = (1.25, 1), borderaxespad = 0)



    g.savefig(f"{root}/eventResults{focus}LayersModalityVsLayers.png")


fig, axes = plt.subplots(1,2, figsize=(15,5))
for i, modality in enumerate(['BOLD','VASO']):
    sns.lineplot(ax = axes[i], data=FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']==modality)], x="volume", y="data", hue='layer', palette='rocket')

    axes[i].set_xlabel('Time (s)', fontsize=24)


    axes[i].set_title(modality, fontsize=24, pad=20)
    axes[i].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')

    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)
    axes[i].set_xticks(ticks)
    axes[i].set_xticklabels(labels,fontsize=24)

    plt.setp(axes[i].get_yticklabels(), fontsize=24)
    axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


plt.rcParams['legend.title_fontsize'] = 20
axes[0].legend().remove()
axes[0].set_ylabel(r'mean $\beta $', fontsize=24)
axes[1].set_ylabel('', fontsize=24)
plt.tight_layout()
plt.savefig('new_layers.png')
plt.show()
