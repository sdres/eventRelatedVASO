import pandas as pd
import numpy as np
import nibabel as nb
import seaborn as sns
import matplotlib.pyplot as plt
import glob
from matplotlib.ticker import MaxNLocator
from matplotlib.ticker import FormatStrFormatter


subList = []
dataList = []
modalityList = []
layerList = []
stimTypeList = []
focusList = []
contrastList = []

root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'
decoRoot = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/deconvolutionAnalysis'



### block profiles

for sub in ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']:
# for sub in ['sub-12']:
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
                    mask = nb.load(f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_11layers_blockStim_layers_equidist.nii').get_fdata()
                    focuses.append(focus)
                except:
                    print(f'{focus} not found')
        print(f'found ROIs for: {focuses}')


        for focus in focuses:
            if not sub == 'sub-07':
                mask = nb.load(f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_11layers_layers_equidist.nii').get_fdata()

            if sub == 'sub-07':
                mask = nb.load(f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_11layers_blockStim_layers_equidist.nii').get_fdata()

            for task in ['blockStimLongTR','blockStim','blockStimVisOnly']:

                blockRuns = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_*_task-{task}_run-00*_cbv.nii.gz'))

                if len(blockRuns)==0:
                    print(f'no runs found for {task}.')
                    continue

                for modality in ['BOLD','VASO']:

                    if 'blockStim' in task:

                        if sub == 'sub-05':
                            data = nb.load(f'{root}/derivatives/{sub}/{ses}/{sub}_{ses}_task-{task}_run-002_{modality}.feat/stats/zstat1_scaled.nii.gz').get_fdata()

                        elif len(blockRuns) == 1:
                            data = nb.load(f'{root}/derivatives/{sub}/{ses}/{sub}_{ses}_task-{task}_run-001_{modality}.feat/stats/zstat1_scaled.nii.gz').get_fdata()

                        else:
                            data = nb.load(f'{root}/derivatives/{sub}/{ses}/{sub}_{ses}_task-{task}_secondLevel_{modality}.gfeat/cope1.feat/stats/zstat1_scaled.nii.gz').get_fdata()

                        for j in range(1,12):  # Compute bin averages
                            layerRoi = mask == j
                            mask_mean = np.mean(data[layerRoi.astype(bool)])

                            subList.append(sub)
                            dataList.append(mask_mean)
                            modalityList.append(modality[:4])
                            layerList.append(j)
                            stimTypeList.append(task)
                            contrastList.append(task)
                            focusList.append(focus)

subList = []
dataList = []
modalityList = []
layerList = []
stimTypeList = []
focusList = []
contrastList = []
### event profiles

for sub in ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']:
# for sub in ['sub-12']:
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



zscores = pd.DataFrame({'subject': subList, 'data': dataList, 'modality': modalityList, 'layer':layerList, 'stimType':stimTypeList, 'contrast':contrastList,'focus':focusList})
zscores



for sub in ['sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-13', 'sub-14']:
    fig, axes = plt.subplots(1, 2,sharex=True,figsize=(15,6))
    for i, modality in enumerate(['VASO', 'BOLD']):
        tmp = zscores.loc[(zscores['subject']==sub)&(zscores['modality']==modality)]
        sns.lineplot(ax=axes[i], data=tmp, x='layer', y='data', hue='contrast')
        axes[i].set_xlabel('WM                                       CSF', fontsize=24)
        axes[i].set_xticks([])
        axes[i].set_title(modality, fontsize=24)

        axes[i].yaxis.set_tick_params(labelsize=20)

    axes[1].legend().remove()
    axes[0].set_ylabel(f'z-score', fontsize=24)
    axes[1].set_ylabel(f'', fontsize=24)
    plt.show()


for focus in ['v1', 's1']:
    fig, axes = plt.subplots(1, 2,sharex=True,figsize=(15,6))
    for i, modality in enumerate(['VASO', 'BOLD']):
        tmp = zscores.loc[(zscores['modality']==modality)&(zscores['focus']==focus)]
        sns.lineplot(ax=axes[i], data=tmp, x='layer', y='data', hue='contrast')
        axes[i].set_xlabel('WM                                       CSF', fontsize=24)
        axes[i].set_xticks([])
        axes[i].set_title(modality, fontsize=24)

        axes[i].yaxis.set_tick_params(labelsize=20)

    axes[1].legend().remove()
    axes[0].set_ylabel(f'z-score', fontsize=24)
    axes[1].set_ylabel(f'', fontsize=24)
    plt.show()

fig, axes = plt.subplots()
tmp = zscores
sns.lineplot(data=tmp, x='layer', y='data', hue='modality')
axes.set_xlabel('WM                                CSF', fontsize=24)
axes.set_xticks([])
axes.set_yticks(range(1,14,2))
axes.set_title('blockStim z-scores across layers', fontsize=24)
axes.yaxis.set_tick_params(labelsize=20)

axes.set_ylabel(f'z-score', fontsize=24)
plt.show()


fig, axes = plt.subplots(1, 2,sharex=True,figsize=(15,6))
fig.subplots_adjust(top=0.8)

for i, modality in enumerate(['VASO', 'BOLD']):

    tmp = zscores.loc[(zscores['modality']==modality)&(zscores['contrast']!='blockStim')]

    sns.lineplot(ax=axes[i], data=tmp, x='layer', y='data', hue='contrast')

    axes[i].set_title(f"{modality[:4]}", fontsize=24)

# axes[0].set_yticks(fontsize=18)
# axes[1].set_yticks(fontsize=18)
axes[0].set_xticks([])
axes[1].set_xticks([])
axes[0].get_legend().remove()
axes[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes[0].set_ylabel(f'z-score', fontsize=24)
axes[1].set_ylabel(f'', fontsize=24)
axes[0].set_xlabel('WM                                     CSF', fontsize=24)
axes[1].set_xlabel('WM                                     CSF', fontsize=24)
plt.savefig(f'{decoRoot}/eventStimProfiles.png', bbox_inches = "tight")
plt.suptitle(f'eventStimRandom layer profiles', fontsize=28, y=0.98)
plt.show()


fig, axes = plt.subplots(figsize=(7.5,6))
fig.subplots_adjust(top=0.8)

tmp = zscores.loc[(zscores['contrast']=='blockStim')]

sns.lineplot(data=tmp, x='layer', y='data', hue='modality')

# axes[0].set_yticks(fontsize=18)
# axes[1].set_yticks(fontsize=18)
axes.set_xticks([])
axes.legend(loc='center left', bbox_to_anchor=(1, 0.5))
axes.set_ylabel(f'z-score', fontsize=24)
axes.set_xlabel('WM                                          CSF', fontsize=24)
plt.suptitle(f'blockStim layer profile', fontsize=28)
plt.savefig(f'{decoRoot}/blockStimProfiles.png', bbox_inches = "tight")

plt.show()


palette1 = {'BOLD': 'tab:orange','VASO': 'tab:blue'}
palette2 = {'BOLD': 'tab:green','VASO': 'tab:blue'}

palettes= [palette1,palette2]

from matplotlib.ticker import MaxNLocator

for i, focus in enumerate(['v1','s1']):

    for stimType in ['blockStim', 'visiotactile']:
        fig, ax = plt.subplots()
        tmp = zscores.loc[(zscores['focus']==focus)&(zscores['contrast']==stimType)]

        sns.lineplot(data=tmp, x='layer', y='data', hue='modality',palette=palettes[i])

        plt.ylabel('z-score', fontsize=24)

        plt.title(f"{stimType} z-scores across layers", fontsize=24, pad=20)
        plt.xlabel('WM                                CSF', fontsize=24)
        plt.xticks([])

        yLimits = ax.get_ylim()

        plt.yticks(range(0,int(yLimits[1]+1),2),fontsize=18)

        plt.legend(loc='upper left', fontsize=20)

        plt.savefig(f'{root}/Group_{focus}_{stimType}_zScoreProfile.png', bbox_inches = "tight")
        plt.show()



palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
}

from matplotlib.ticker import MaxNLocator

for focus in ['v1']:
    for modality in ['BOLD', 'VASO']:
        fig, ax = plt.subplots()
        tmp = zscores.loc[(zscores['focus']==focus)&(zscores['modality']==modality)]
        # for contrast in ['visiotactile', 'blockStim']:
            # sns.lineplot(data=tmp.loc[zscores['contrast']==contrast], x='layer', y='data', hue='contrast')
        sns.lineplot(data=tmp, x='layer', y='data', hue='contrast')

        plt.ylabel('z-score', fontsize=24)

        plt.title(f"{modality} z-scores across layers", fontsize=24, pad=20)
        plt.xlabel('WM                                CSF', fontsize=24)
        plt.xticks([])

        yLimits = ax.get_ylim()
        plt.ylim(0,yLimits[1]+1)

        plt.yticks(range(0,int(yLimits[1]+1),2),fontsize=18)

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

        plt.legend(loc='upper left', fontsize=20)


        # plt.savefig(f'{root}/Group_{stimType}_zScoreProfile.png', bbox_inches = "tight")
        plt.show()
