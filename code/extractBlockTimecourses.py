import numpy as np
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import os

subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']


root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

modalities = ['BOLD', 'VASO']

blockResults = {}
for focus in ['v1','s1']:
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
        print(f'{sub} has {len(sessions)} sessions.')

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

for sub in subs:
    for focus, cmap in zip(['v1','s1'],palettes):
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


for focus, cmap in zip(['v1','s1'],palettes):
    fig, ax = plt.subplots(figsize=(7,5))
    sns.lineplot(data=blockData.loc[(blockData['focus']==focus)], x='x', y='data', hue='modality', palette = cmap)


    plt.axvspan(4, 4+(30/tr), color='grey', alpha=0.2, lw=0, label = 'stimulation on')
    plt.ylabel('% signal change', fontsize=24)
    plt.xlabel('Time (s)', fontsize=24)
    plt.title(f"Group Response Timecourse", fontsize=24, pad=20)
    plt.legend(loc='lower center', fontsize=14)


    values = (np.arange(-4,len(blockData['x'].unique())-4,4)*tr).round().astype(int)
    spacing = np.arange(0,len(blockData['x'].unique()),4)

    plt.xticks(spacing,values, fontsize=18)
    plt.yticks(fontsize=18)
    plt.savefig(f'{root}/group_{focus}_BlockResults_ts.png', bbox_inches = "tight")
    plt.show()



subList = []
runList = []
modalityList = []
layerList = []
timepointList = []
dataList = []
layers = {'1':'deep','2':'middle','3':'superficial'}

for sub in ['sub-14']:
    runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-block*_run-00*_cbv.nii.gz'))

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
            for timepoint in range(1,41):
                dataFile = f'{root}/derivatives/{sub}/{ses}/{base}_{modality}_layers_FIR.feat/stats/pe{timepoint}.nii.gz'
                dataNii = nb.load(dataFile)
                data = dataNii.get_fdata()
                for layer in layers.keys():
                    subList.append(sub)
                    runList.append(base)
                    modalityList.append(modality)
                    layerList.append(layers[layer])
                    timepointList.append(timepoint)
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

FIRdata = pd.DataFrame({'subject':subList, 'run':runList, 'layer':layerList, 'modality':modalityList, 'data':dataList, 'volume':timepointList})

for sub in ['sub-14']:
    fig, axes = plt.subplots(1,2)
    for i, modality in enumerate(['BOLD', 'VASO']):
        tmp = FIRdata.loc[(FIRdata['modality']==modality)&(FIRdata['subject']==sub)]
        sns.lineplot(ax = axes[i], data=tmp, x="volume", y="data", hue='layer')
    plt.suptitle(sub, fontsize=20)
    plt.show()
