import numpy as np
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import os
from matplotlib.ticker import FormatStrFormatter

subs = ['sub-05']
ses = 'ses-001'
root = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

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
            outFolder = f'{root}/derivatives/{sub}/{ses}/old/upsampledFunctional'

            if sub == 'sub-09' and focus == 's1Focus' and ses=='ses-002':
                continue

            runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-eventSti*_run-00*_cbv.nii.gz'))

            for run in runs[1:]:
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

        for run in runs[1:]:

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

for modality in modalities:
    sns.lineplot(data=layerEventData.loc[layerEventData['modality']==modality], x="x", y="data", hue='layer')
    plt.title(modality)
    plt.show()

for modality in modalities:
    sns.lineplot(data=layerEventData.loc[layerEventData['modality']==modality], x="x", y="data", hue='layer')
    plt.title(modality)
    plt.show()

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
# for sub in ['sub-05']:
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


FIRdata.to_csv('../results/FIR_results.csv',index=False)
FIRdata = pd.read_csv('/Users/sebastiandresbach/git/eventRelatedVASO/results/FIR_results.csv')

# for sub in ['sub-06', 'sub-07','sub-08', 'sub-09', 'sub-13', 'sub-14']:
for sub in ['sub-14']:
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
    'BOLD': 'tab:orange',
    'VASO': 'tab:green'}

palettes = [v1Palette,s1Palette]

for focus, cmap in zip(['v1','s1'],palettes):


    fig, ax  = plt.subplots()
    sns.lineplot(data=FIRdata.loc[FIRdata['focus']==focus], x="volume", y="data", hue='modality',palette=cmap)
    plt.title('Group Finite Impulse Response', fontsize=24, pad=20)
    plt.ylabel(r'mean $ \beta $', fontsize=24)
    yLimits = ax.get_ylim()

    plt.ylim(0,yLimits[1])
    plt.yticks(range(-1,int(yLimits[1])+1),fontsize=16)

    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)

    plt.xticks(ticks[::2], labels[::2], fontsize=16)
    plt.xlabel('Time [s]', fontsize=24)

    plt.axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')
    plt.legend(loc='upper right', fontsize=14)
    plt.savefig(f'/Users/sebastiandresbach/git/eventRelatedVASO/results/Group_{focus}_FIR.png', bbox_inches = "tight")

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
        tmp = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['focus']==focus)]
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



    # g.savefig(f"{root}/eventResults{focus}LayersModalityVsLayers.png")


palette2 = {
    'superficial': '#1F77B4',
    'middle': '#726DBA',
    'deep': '#C65293'}

palette1 = {
    'superficial': '#FF7F0E',
    'middle': '#AA3900',
    'deep': '#5F0000'}

palettes = [palette1,palette2]

fig, axes = plt.subplots(1,2, figsize=(15,5))

for i, (modality,palette) in enumerate(zip(['BOLD','VASO'],palettes)):
    sns.lineplot(ax = axes[i], data=FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']==modality)], x="volume", y="data", hue='layer', palette='rocket')

    axes[i].set_xlabel('Time [s]', fontsize=24)


    axes[i].set_title(modality, fontsize=24, pad=20)
    axes[i].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')

    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)
    axes[i].set_xticks(ticks[::2])
    axes[i].set_xticklabels(labels[::2],fontsize=24)

    plt.setp(axes[i].get_yticklabels(), fontsize=24)
    axes[i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


axes[1].legend().remove()
axes[0].legend(loc = 'upper right',fontsize=14)
axes[0].set_ylabel(r'mean $ \beta $', fontsize=24)
axes[1].set_ylabel('', fontsize=24)
plt.tight_layout()
plt.savefig('/Users/sebastiandresbach/git/eventRelatedVASO/results/new_layers.png')
plt.show()




#############################################
########### FIR long vs shortITI ############
#############################################


layers = {'1':'deep','2':'middle','3':'superficial'}

subList = []
runList = []
modalityList = []
layerList = []
timepointList = []
dataList = []
focusList = []
longShortList = []

# for sub in ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-12','sub-13', 'sub-14']:
for sub in ['sub-05','sub-06', 'sub-07','sub-08','sub-09']:
    runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-event*_run-00*_cbv.nii.gz'))
    # for focus in ['v1', 's1']:
    for focus in ['v1']:
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
                for type in ['longITI', 'shortITI']:


                    for timepoint in range(1,11):
                        if type == 'longITI':
                            pe = timepoint
                        if type == 'shortITI':
                            pe = timepoint+10

                        try:
                            dataFile = f'{root}/derivatives/{sub}/{ses}/{base}_{focus}_{modality}_layers_FIR_longVsShortITI.feat/stats/pe{pe}.nii.gz'
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
                            longShortList.append(type)
                            if modality =='BOLD':
                                dataList.append(data[int(layer)-1])
                            if modality =='VASO':
                                dataList.append(-data[int(layer)-1])



FIRdata = pd.DataFrame({'subject':subList, 'run':runList, 'layer':layerList, 'modality':modalityList, 'data':dataList, 'volume':timepointList, 'focus':focusList, 'ITI':longShortList})




fig, axes = plt.subplots(2,2, figsize=(15,10))
for j, type in enumerate(['longITI', 'shortITI']):
    for i, modality in enumerate(['BOLD','VASO']):
        sns.lineplot(ax = axes[j,i], data=FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']==modality)&(FIRdata['ITI']==type)], x="volume", y="data", hue='layer', palette='rocket')

        axes[j,i].set_xlabel('Time (s)', fontsize=24)


        axes[j,i].set_title(modality, fontsize=24, pad=20)
        axes[j,i].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')

        ticks = range(0,11)
        labels = (np.arange(0,11)*1.3).round(decimals=1)
        axes[j,i].set_xticks(ticks)
        axes[j,i].set_xticklabels(labels,fontsize=24,rotation=45)

        plt.setp(axes[j,i].get_yticklabels(), fontsize=24)
        axes[j,i].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        axes[j,i].axhline(0,linestyle='--',color='black')

    plt.rcParams['legend.title_fontsize'] = 20
    axes[j,0].legend().remove()
    axes[j,0].set_ylabel(r'% signal change', fontsize=24)
    axes[j,1].set_ylabel('', fontsize=24)

plt.tight_layout()
# plt.savefig('../results/new_layers.png')
plt.show()



fig, axes = plt.subplots(3,1, figsize=(8,15))
for j, layer in enumerate(['superficial', 'middle','deep']):
    sns.lineplot(ax = axes[j], data=FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']=='VASO')&(FIRdata['layer']==layer)], x="volume", y="data", hue='ITI', palette='rocket')

    axes[j].set_xlabel('Time (s)', fontsize=24)


    axes[j].set_title(layer, fontsize=24, pad=20)
    axes[j].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')

    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)
    axes[j].set_xticks(ticks)
    axes[j].set_xticklabels(labels,fontsize=24,rotation=45)

    plt.setp(axes[j].get_yticklabels(), fontsize=24)
    axes[j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[j].axhline(0,linestyle='--',color='black')

    plt.rcParams['legend.title_fontsize'] = 20
axes[1].set_ylabel(r'VASO % signal change', fontsize=24)
axes[0].set_ylabel('', fontsize=24)
axes[2].set_ylabel('', fontsize=24)

plt.tight_layout()
plt.savefig('../results/shortVsLongITI_allLayers.png')
plt.show()

fig, axes = plt.subplots(1,1, figsize=(8,15))
for j, layer in enumerate(['superficial']):
    sns.lineplot(ax = axes[j], data=FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']=='VASO')&(FIRdata['layer']==layer)], x="volume", y="data", hue='ITI', palette='rocket')

    axes[j].set_xlabel('Time (s)', fontsize=24)


    axes[j].set_title(layer, fontsize=24, pad=20)
    axes[j].axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')

    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)
    axes[j].set_xticks(ticks)
    axes[j].set_xticklabels(labels,fontsize=24,rotation=45)

    plt.setp(axes[j].get_yticklabels(), fontsize=24)
    axes[j].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    axes[j].axhline(0,linestyle='--',color='black')

    plt.rcParams['legend.title_fontsize'] = 20
    axes[0].set_ylabel(r'VASO % signal change', fontsize=24)


plt.tight_layout()
plt.savefig('../results/shortVsLongITI_superficial.png')
plt.show()



v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}
s1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:green'}

palettes = [v1Palette,s1Palette]

### BLACK BG

plt.style.use('dark_background')
plt.rcParams["font.family"] = "Times New Roman"
for focus, cmap in zip(['v1','s1'],palettes):


    fig, ax  = plt.subplots()
    sns.lineplot(data=FIRdata.loc[FIRdata['focus']==focus], x="volume", y="data", hue='modality',palette=cmap)
    plt.title('', fontsize=24, pad=20)
    plt.ylabel(r'signal change [%]', fontsize=24)
    yLimits = ax.get_ylim()

    plt.ylim(0,yLimits[1])
    plt.yticks(range(-1,int(yLimits[1])+1),fontsize=16)

    # ticks = range(0,11)
    # labels = (np.arange(0,11)*1.3).round(decimals=1)
    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)
    for k,label in enumerate(labels):
        if (label - int(label) == 0):
            labels[k] = int(label)

    ## reduce input


    # plt.xticks(ticks, labels, fontsize=16, rotation=45)
    ax.set_xticks(ticks[::2])
    ax.set_xticklabels(labels[::2],fontsize=16)


    plt.xlabel('Time [s]', fontsize=24)

    plt.axvspan(0, 2/1.3, color='#e5e5e5', alpha=0.2, lw=0, label = 'stimulation on')
    ax.axhline(0,linestyle='--',color='white')

    plt.legend(loc='upper right', fontsize=13).remove()
    plt.savefig(f'../results/Group_{focus}_FIR_manuscript.png', bbox_inches = "tight")

    plt.show()


## reduce input
v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}
s1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:green'}

palettes = [v1Palette,s1Palette]

fig, axes = plt.subplots(3,2, figsize=(10,15))

# for j, layer in enumerate(['superficial','middle','deep']):
for i, (modality,palette) in enumerate(zip(['BOLD','VASO'],['tab:orange','tab:blue'])):
    axes[0,i].set_title(modality, fontsize=24, pad=20)

    for j, layer in enumerate(['superficial','middle','deep']):

        sns.lineplot(ax = axes[j,i], data=FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']==modality)&(FIRdata['layer']==layer)], x="volume", y="data", color=palette, linewidth=2)

        vasoLim = [-1.5, 9]
        boldLim = [-1.5,9]

        if modality == 'VASO':
            axes[j,i].set_ylim(vasoLim[0],vasoLim[1])
        if modality == 'BOLD':
            axes[j,i].set_ylim(boldLim[0],boldLim[1])
        axes[j,i].axhline(0,linestyle='--',color='white')
        axes[j,i].axvline(4,linestyle='--',color='white')

        axes[j,i].set_xlabel('Time [s]', fontsize=24)

        axes[j,i].axvspan(0, 2/1.3, color='white', alpha=0.2, lw=0, label = 'stimulation')
        #
        ticks = range(0,11)
        labels = (np.arange(0,11)*1.3).round(decimals=1)
        for k,label in enumerate(labels):
            if (label - int(label) == 0):
                labels[k] = int(label)

        ## reduce input

        axes[j,i].set_xticks(ticks[::2])
        axes[j,i].set_xticklabels(labels[::2],fontsize=20)


for j in range(3):
    axes[j,1].set_ylabel('', fontsize=24)
    axes[j,0].set_ylabel('', fontsize=24)


    ax2 = axes[j,1].twinx()
    ax2.set_ylabel(['superficial','middle','deep'][j], fontsize=24)
    ax2.set_yticks([])
    axes[j,1].set_yticks([])


    plt.setp(axes[j,0].get_yticklabels(), fontsize=20)
    axes[j,0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.setp(axes[j,1].get_yticklabels(), fontsize=20)
    axes[j,1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))


# show ylabel for middle row only
axes[1,0].set_ylabel('signal change [%]', fontsize=24)

# remove x ticks and labels except for lowest row
for j in range(2):
    axes[j,0].set_xlabel('', fontsize=24)
    axes[j,1].set_xlabel('', fontsize=24)
    axes[j,0].set_xticks([])
    axes[j,0].set_xticklabels([],fontsize=24,rotation=45)
    axes[j,1].set_xticks([])
    axes[j,1].set_xticklabels([],fontsize=24,rotation=45)

axes[-1,-1].legend(fontsize=14)
plt.tight_layout()
plt.savefig('../results/layersInPanels.png')
plt.show()






# Normalize response peak

FIRdata

tmpVASO = FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']=='VASO')]
# find peak
maxtp=0
maxval = 0
for timepoint in tmpVASO['volume'].unique():
    tmp = np.mean(tmpVASO['data'].loc[tmpVASO['volume']==timepoint])
    if tmp > maxval:
        maxtp=timepoint
        maxval = tmp



tmpVASO['dataNorm'] = np.array(tmpVASO['data'])/np.mean(tmpVASO['data'].loc[tmpVASO['volume']==maxtp])



tmpBOLD = FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']=='BOLD')]
# find peak
maxtp=0
maxval = 0
for timepoint in tmpBOLD['volume'].unique():
    tmp = np.mean(tmpBOLD['data'].loc[tmpBOLD['volume']==timepoint])
    if tmp > maxval:
        maxtp=timepoint
        maxval = tmp

tmpBOLD['dataNorm'] = tmpBOLD['data']/np.mean(tmpBOLD['data'].loc[tmpBOLD['volume']==maxtp])



# tmpBOLD = FIRdata.loc[(FIRdata['focus']=='v1')&(FIRdata['modality']=='BOLD')]
# tmpBOLD['data'] = tmpBOLD['data']/np.amax(tmpBOLD['data'])

tmp = tmpVASO.append(tmpBOLD)



fig, ax  = plt.subplots()
sns.lineplot(data=tmpVASO, x="volume", y="dataNorm",label='VASO')
sns.lineplot(data=tmpBOLD, x="volume", y="dataNorm",label='BOLD')
plt.legend()
plt.show()



tmp['volume']=tmp['volume']-1
v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}
s1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:green'}


palettes = [v1Palette,s1Palette]


for focus in ['v1']:


    fig, ax  = plt.subplots()
    sns.lineplot(data=tmp.loc[tmp['focus']==focus], x="volume", y="dataNorm", hue='modality',palette=v1Palette)
    plt.title('Group Finite Impulse Response', fontsize=24, pad=20)
    plt.ylabel(r'% signal change', fontsize=24)


    yLimits = ax.get_ylim()

    plt.ylim(-0.2,yLimits[1])
    plt.yticks(np.linspace(-0.2,1.2,8).round(decimals=2),fontsize=16)

    # ax.tick_params(axis='y', labelsize= 16)

    ticks = range(0,11)
    labels = (np.arange(0,11)*1.3).round(decimals=1)

    plt.xticks(ticks, labels, fontsize=16, rotation=45)
    plt.xlabel('Time (s)', fontsize=24)

    plt.axvspan(0, 2/1.3, color='grey', alpha=0.2, lw=0, label = 'stimulation on')
    plt.legend(loc='upper right', fontsize=13)
    plt.savefig(f'{root}/Group_{focus}_FIR_norm.png', bbox_inches = "tight")

    plt.show()







###################################
########### FIR random ############
###################################


layers = {'1':'deep','2':'middle','3':'superficial'}

subList = []
runList = []
modalityList = []
layerList = []
timepointList = []
dataList = []
focusList = []
conditionList = []

for sub in ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-12','sub-13', 'sub-14']:
# for sub in ['sub-05']:
    runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-event*_run-00*_cbv.nii.gz'))
    # for focus in ['v1', 's1']:
    for focus in ['v1']:
        for run in runs:

            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(base)

            if 'Random' in base:
                print('processing...')
            else:
                print('skipping...')
                continue

            if 'ses-001' in base:
                ses = 'ses-001'
            else:
                ses= 'ses-002'

            if sub == 'sub-09' and '4' in run:
                print('skipping')
                continue

            for modality in ['BOLD', 'VASO']:
                print(modality)

                timepoint = 0

                for pe in range(1,21):


                    if pe == 11:
                        timepoint = 0

                    timepoint = timepoint + 1

                    print(f'PE :{pe}')
                    print(f'timepoint :{timepoint}')

                    try:
                        dataFile = f'{root}/derivatives/{sub}/{ses}/{base}_{focus}_{modality}_layers_FIR.feat/stats/pe{pe}.nii.gz'
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


                        if pe <= 10:
                            conditionList.append('visual')
                            print('visual')
                        if pe >= 11:
                            conditionList.append('visiotactile')
                            print('visiotactile')
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

FIRdata = pd.DataFrame({'subject':subList, 'run':runList, 'layer':layerList, 'modality':modalityList, 'data':dataList, 'volume':timepointList, 'focus':focusList, 'condition':conditionList})

FIRdata.to_csv('../results/FIRdataRandom.csv', index=False)

for modality in ['BOLD', 'VASO']:
    fig, axes = plt.subplots(1,3)
    fig.subplots_adjust(top=0.8)
    for i, layer in enumerate(['superficial', 'middle', 'deep']):


        data = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['layer']==layer)]

        sns.lineplot(ax=axes[i], data=data, x='volume', y='data', hue='condition')

        axes[i].axvspan(0, 2/tr, color='grey', alpha=0.2, lw=0)
        axes[i].set_xlabel('Timepoints', fontsize=20)
        axes[i].set_title(layer)

    axes[0].get_legend().remove()
    axes[1].get_legend().remove()
    axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

    axes[0].set_ylabel(r'mean $\beta$', fontsize=20)
    axes[1].set_ylabel('', fontsize=24)
    plt.suptitle(f'{modality}', fontsize=28, y=0.98)
    # plt.savefig(f'{decoRoot}/FIR_{modality}_layers.png', bbox_inches = "tight")
    plt.show()



tr = 1.3
for sub in FIRdata['subject'].unique():

    for modality in ['BOLD', 'VASO']:
        fig, axes = plt.subplots(1,3)
        fig.subplots_adjust(top=0.8)
        for i, layer in enumerate(['superficial', 'middle', 'deep']):


            data = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['layer']==layer)&(FIRdata['subject']==sub)]

            sns.lineplot(ax=axes[i], data=data, x='volume', y='data', hue='condition')

            axes[i].axvspan(0, 2/tr, color='grey', alpha=0.2, lw=0)
            axes[i].set_xlabel('Timepoints', fontsize=20)
            axes[i].set_title(layer)

        axes[0].get_legend().remove()
        axes[1].get_legend().remove()
        axes[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))

        axes[0].set_ylabel(r'mean $\beta$', fontsize=20)
        axes[1].set_ylabel('', fontsize=24)
        plt.suptitle(f'{sub}_{modality}', fontsize=28, y=0.98)
        # plt.savefig(f'{decoRoot}/FIR_{modality}_layers.png', bbox_inches = "tight")
        plt.show()


 # FIRdata.to_csv('../results/FIR_results.csv',index=False)
