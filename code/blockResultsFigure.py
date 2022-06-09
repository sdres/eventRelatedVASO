import numpy as np
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import os

subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']
subs = ['sub-05']


root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

modalities = ['BOLD', 'VASO']

blockResults = {}
# for focus in ['v1','s1']:
for focus in ['v1']:
    print((focus))

    blockResults[focus] = {}
    for sub in subs:
        print(sub)

        blockRuns = sorted(glob.glob(f'{root}/{sub}/*/func/{sub}_*_task-blockStim*_run-00*_cbv.nii.gz'))

        sessions = []
        for run in blockRuns:
            if 'ses-001' in run:
                sessions.append('ses-001')
            if 'ses-002' in run:
                sessions.append('ses-002')
        sessions = set(sessions)
        print(f'{sub} has {len(sessions)} session(s).')

        for ses in sessions:
            print(ses)
            outFolder = f'{root}/derivatives/{sub}/{ses}/upsampledFunctional'

            if not sub == 'sub-07':
                try:
                    mask = nb.load(f'{outFolder}/{sub}_{focus}_3layers_layers_equidist.nii.gz').get_fdata()
                except:
                    print(f'no mask found for {focus}.')
                    continue
            if sub == 'sub-07':
                try:
                    mask = nb.load(f'{outFolder}/{sub}_{focus}_3layers_layers_blockStim_equidist.nii.gz').get_fdata()
                except:
                    print(f'no mask found for {focus}.')
                    continue

            blockResults[focus][sub] = {}



            runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-blockSti*_run-00*_cbv.nii.gz'))

            for run in runs:
                # exclude runs with long TR because of different number of timepoints
                if 'Long' in run:
                    continue
                if 'Random' in run:
                    continue

                base = os.path.basename(run).rsplit('.', 2)[0][:-4]
                print(f'processing {base}')
                blockResults[focus][sub][base] = {}


                for modality in modalities:
                    blockResults[focus][sub][base][modality] = {}

                    print(modality)

                    run = f'{outFolder}/{base}_{focus}_{modality}.nii.gz'

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

                    events = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/events/{base}.txt', sep = ' ', names = ['start','duration','trialType'])

                    # Load information on run-wise motion
                    FDs = pd.read_csv(f'{root}/derivatives/{sub}/{ses}/motionParameters/{base}_FDs.csv')

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
                        mask_mean = np.mean(data[:, :, :]
                                            [mask.astype(bool)])

                        # truncate motion data to event
                        tmp = FDs.loc[(FDs['volume']>=(onset-2)/2)&(FDs['volume']<=(offset+2)/2)]

                        if not (tmp['FD']>=2).any():
                            blockResults[focus][sub][base][modality][f'trial {i}'] = np.mean((((data[:, :, :, int(onset-4):int(offset+ 8)][mask.astype(bool)]) / mask_mean)- 1) * 100,axis=0)


subList = []
dataList = []
xList = []
modalityList = []
trialList = []
runList = []
focusList = []

for focus in ['s1', 'v1']:
    print(focus)

    for sub in subs:
        print(sub)
        runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-001_task-blockStim_run-00*_cbv.nii.gz'))


        for run in runs:

            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(base)

            try:
                for modality in modalities:

                    subTrials = []
                    for key, value in blockResults[focus][sub][base][modality].items():
                        subTrials.append(key)

                    for trial in subTrials:

                        for n in range(len(blockResults[focus][sub][base][modality][trial][0])):

                            if modality == "BOLD":
                                dataList.append(blockResults[focus][sub][base][modality][trial][0][n])

                            if modality == "VASO":
                                dataList.append(-blockResults[focus][sub][base][modality][trial][0][n])

                            modalityList.append(modality)
                            trialList.append(trial)
                            runList.append(base)
                            xList.append(n)
                            subList.append(sub)
                            focusList.append(focus)
            except:
                print('data not available')

blockData = pd.DataFrame({'subject': subList,'x':xList, 'data': dataList, 'modality': modalityList, 'trial': trialList, 'run':runList, 'focus':focusList})

v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}
s1Palette = {
    'BOLD': 'tab:green',
    'VASO': 'tab:blue'}

palettes = [v1Palette,s1Palette]

plt.style.use('dark_background')

for focus in ['v1','s1']:
    fig, ax = plt.subplots(figsize=(7,5))
    sns.lineplot(data=blockData.loc[(blockData['focus']==focus)], x='x', y='data', hue='modality', palette = v1Palette)


    plt.axvspan(4, 4+(30/tr), color='grey', alpha=0.2, lw=0, label = 'stimulation')
    plt.ylabel('signal change [%]', fontsize=24)
    plt.xlabel('Time [s]', fontsize=24)
    # plt.title(f"Group Response Timecourse", fontsize=24, pad=20)
    plt.legend(loc='lower center', fontsize=14)


    values = (np.arange(-4,len(blockData['x'].unique())-4,4)*tr).round().astype(int)
    spacing = np.arange(0,len(blockData['x'].unique()),4)

    plt.xticks(spacing,values, fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(f'../results/group_{focus}_BlockResults_ts.png', bbox_inches = "tight")
    plt.show()




subList = []
dataList = []
modalityList = []
layerList = []
stimTypeList = []
focusList = []
contrastList = []
runList = []


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
import matplotlib.gridspec as gridspec

zscores = pd.DataFrame({'subject': subList, 'data': dataList, 'modality': modalityList, 'layer':layerList, 'stimType':stimTypeList, 'contrast':contrastList,'focus':focusList, 'runType':runList})



for sub in subs:
    for focus, cmap in zip(['v1'],palettes):
        sns.lineplot(data=blockData.loc[(blockData['subject']==sub)&(blockData['focus']==focus)], x='x', y='data', hue='modality', palette = cmap)


        plt.axvspan(4, 4+(30/tr), color='grey', alpha=0.2, lw=0, label = 'stimulation on')
        plt.ylabel('% signal change', fontsize=24)
        plt.xlabel('Time (s)', fontsize=24)
        plt.title(f"{sub} Event-Related Average", fontsize=24, pad=20)
        plt.legend(loc='lower center', fontsize=14)


        values = (np.arange(-4,len(blockData['x'].unique())-4,4)*tr).round().astype(int)
        spacing = np.arange(0,len(blockData['x'].unique()),4)

        plt.xticks(spacing,values, fontsize=18)
        plt.yticks(fontsize=18)
        # plt.savefig(f'{root}/{sub}_{focus}_BlockResults.png', bbox_inches = "tight")
        plt.show()

for focus in ['v1']:
    fig, (ax1,ax2) = plt.subplots(2,1,figsize=(7.5,10))


    sns.lineplot(ax=ax1, data=blockData.loc[(blockData['focus']==focus)], x='x', y='data', hue='modality', palette = v1Palette)


    ax1.axvspan(4, 4+(30/tr), color='grey', alpha=0.2, lw=0, label = 'stimulation')
    ax1.set_ylabel('signal change [%]', fontsize=24)
    ax1.set_xlabel('Time [s]', fontsize=24)
    ax1.legend(loc='lower center', fontsize=14)

    values = (np.arange(-4,len(blockData['x'].unique())-4,4)*tr).round().astype(int)
    spacing = np.arange(0,len(blockData['x'].unique()),4)
    ax1.set_xticks(spacing,values, fontsize=18)

    ax1.tick_params(axis='y', labelsize=18)
    # ax1.legend().remove()



    tmp = zscores.loc[(zscores['focus']==focus)&(zscores['stimType']== 'blockStim')]
    sns.lineplot(ax=ax2, data=tmp, x='layer', y='data', hue='modality',palette=v1Palette)

    ax2.set_ylabel(f'z-score', fontsize=24)
    ax2.set_xticks([])

    ax2.set_xlabel('WM                                                 CSF', fontsize=24)
    ax2.tick_params(axis='y', labelsize=18)

    ax2.legend().remove()

    fig.tight_layout()


    # lines = []
    # labels = []
    #
    # for ax in fig.axes:
    #     axLine, axLabel = ax.get_legend_handles_labels()
    #     lines.extend(axLine)
    #     labels.extend(axLabel)
    #
    #
    # fig.legend(lines[:2], labels[:2],
    #            loc = 'center right', fontsize=14)

    plt.savefig(f'../results/group_{focus}_BlockResults.png', bbox_inches = "tight")
    plt.show()
