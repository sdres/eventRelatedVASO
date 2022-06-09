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
runList = []

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

                if not task == 'blockStimVisOnly':
                    contrast = 'visiotactile'
                else:
                    contrast = 'visual'

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
                            contrastList.append(contrast)
                            focusList.append(focus)
                            runList.append('blockStim')


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
zscores


zscores['contrast'].unique()

zscores.loc[zscores['subject']=='sub-09']



VASOcmap = {
    'eventStim': '#1f77b4',
    'blockStim': '#00bbe8'}
BOLDcmap = {
    'eventStim': '#ff0600',
    'blockStim': '#ff9c00'}

palettes = [VASOcmap,BOLDcmap]
from matplotlib.ticker import MaxNLocator

# for focus, cmap in zip(['s1', 'v1'],['Dark2', 'tab10']):
for focus in ['v1']:
    for modality,cmap in zip(['VASO', 'BOLD'],palettes):
        fig, ax = plt.subplots()
        tmp = zscores.loc[(zscores['modality']==modality)&(zscores['focus']==focus)&(zscores['contrast']=='visiotactile')&(zscores['stimType']!='blockStimLongTR')]

        sns.lineplot(data=tmp, x='layer', y='data', hue='runType',palette=cmap)


        plt.ylabel(f'{focus[:2]} z-score', fontsize=24)

        plt.title(f"{modality[:4]} z-scores across layers", fontsize=24, pad=20)
        plt.xlabel('WM                                CSF', fontsize=24)
        plt.xticks([])

        plt.yticks(fontsize=18)

        ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

        plt.legend(loc='upper left')

        yLimits = ax.get_ylim()
        plt.ylim(0,yLimits[1])
        plt.legend(loc='upper left',fontsize=20)

        plt.savefig(f'../results/Group_{focus}_{modality}_zScoreProfile.png', bbox_inches = "tight")
        plt.show()


palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
}
# for focus, cmap in zip(['s1', 'v1'],['Dark2', 'tab10']):
for focus in ['v1']:
    fig, ax = plt.subplots()
    tmp = zscores.loc[(zscores['focus']==focus)&(zscores['contrast']=='visiotactile')&(zscores['runType']== 'eventStim')]
    # zscores.loc[(zscores['focus']==focus)&(zscores['contrast']=='visiotactile')&(zscores['stimType']!= 'blockStimVisOnly')&(zscores['stimType']!='blockStimLongTR')&(zscores['stimType']!='blockStim')]
    sns.lineplot(data=tmp, x='layer', y='data', hue='modality',palette=palette)


    plt.ylabel(f'{focus[:2]} z-score', fontsize=24)

    plt.title(f"eventStim z-scores across layers", fontsize=24, pad=20)
    plt.xlabel('WM                                CSF', fontsize=24)
    plt.xticks([])

    plt.yticks(fontsize=18)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    plt.legend(loc='upper left')

    yLimits = ax.get_ylim()
    plt.ylim(0,yLimits[1])
    plt.legend(loc='upper left',fontsize=20)

    plt.savefig(f'../results/Group_{focus}_eventStim_zScoreProfile.png', bbox_inches = "tight")
    plt.show()




subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']

for sub in ['sub-14','sub-08']:
    fig, ax = plt.subplots()
    for focus in ['v1']:
        for modality,cmap in zip(['VASO', 'BOLD'],palettes):
            # for stimType in ['blockStim', 'eventStim']:
            # fig, ax = plt.subplots()
            # tmp = zscores.loc[(zscores['subject']==sub)&(zscores['focus']==focus)&(zscores['contrast']=='visiotactile')&(zscores['stimType']==stimType)]
            tmp = zscores.loc[(zscores['modality']==modality)&(zscores['focus']==focus)&(zscores['contrast']=='visiotactile')&(zscores['stimType']!= 'blockStimVisOnly')&(zscores['stimType']!='blockStimLongTR')&(zscores['subject']==sub)]
            sns.lineplot(data=tmp, x='layer', y='data', hue='runType',palette=cmap)


            plt.ylabel(f'{focus[:2]} z-score', fontsize=24)

            plt.title(f"{sub} {modality} z-scores across layers", fontsize=24, pad=20)
            plt.xlabel('WM                                CSF', fontsize=24)
            plt.xticks([])

            plt.yticks(fontsize=18)

            ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            plt.legend(loc='upper left')

            yLimits = ax.get_ylim()
            # plt.ylim(0,yLimits[1])
            plt.legend(loc='upper left',fontsize=20)

            plt.savefig(f'../results/{sub}_{focus}_{modality}_zScoreProfile.png', bbox_inches = "tight")
            plt.show()



zscores['stimType'].unique()


plt.style.use('dark_background')

palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'
}
for focus, cmap in zip(['s1', 'v1'],['Dark2', 'tab10']):
# for focus in ['v1']:
    fig, ax = plt.subplots(figsize=(7,5))
    tmp = zscores.loc[(zscores['focus']==focus)&(zscores['stimType']== 'blockStim')]
    # zscores.loc[(zscores['focus']==focus)&(zscores['contrast']=='visiotactile')&(zscores['stimType']!= 'blockStimVisOnly')&(zscores['stimType']!='blockStimLongTR')&(zscores['stimType']!='blockStim')]
    sns.lineplot(data=tmp, x='layer', y='data', hue='modality',palette=palette)


    # plt.ylabel(f'z-score', fontsize=24)
    #
    # plt.title(f"", fontsize=24, pad=20)
    # plt.xlabel('WM                                CSF', fontsize=24)
    # plt.xticks([])
    #
    # plt.yticks(fontsize=18)

    # ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # yLimits = ax.get_ylim()
    # plt.ylim(0,yLimits[1])
    plt.legend().remove()

    plt.ylabel(f'', fontsize=24)

    plt.title(f"", fontsize=24, pad=20)
    plt.xlabel('', fontsize=24)
    plt.xticks([])

    plt.yticks([])


    plt.savefig(f'../results/Group_{focus}_blockStim_zScoreProfile.png', bbox_inches = "tight")
    plt.show()
