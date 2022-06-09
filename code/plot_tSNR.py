import numpy as np
import nibabel as nb
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


plt.style.use('dark_background')


root = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']
# subs = ['sub-09','sub-11']

modalities = ['BOLD', 'VASO']

subList = []
modalityList = []
tSNRList = []
meanList = []
voxelList = []
runList = []
focusList = []
kurtList = []
skewList = []


for sub in subs:
    print(sub)

    runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-*_cbv.nii.gz'))

    for run in runs:

        base = os.path.basename(run).rsplit('.', 2)[0][:-4]

        if 'ses-001' in base:
            ses = 'ses-001'
        if 'ses-002' in base:
            ses = 'ses-002'
        print(ses)
        for modality in modalities:
            for focus in ['v1', 's1']:
                try:
                    if sub == 'sub-07':
                        if 'block' in base:
                            maskFile = f'{root}/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_3layers_blockStim_layers_equidist.nii'
                        else:
                            maskFile = f'{root}/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_3layers_eventStim_layers_equidist.nii'
                    else:
                        maskFile = f'{root}/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_3layers_layers_equidist.nii'

                    maskNii = nb.load(maskFile)
                    maskData = maskNii.get_fdata()
                    idxMask = maskData > 0
                except:
                    print(f'no mask found for {focus}')
                    continue

                tSNRFile = f'{root}/derivatives/{sub}/{ses}/{base}_{modality}_tSNR_scaled.nii.gz'
                tSNRNii = nb.load(tSNRFile)
                tSNRData = tSNRNii.get_fdata()

                meanFile = f'{root}/derivatives/{sub}/{ses}/{base}_{modality}_mean_scaled.nii.gz'
                meanNii = nb.load(meanFile)
                meanData = meanNii.get_fdata()

                skewFile = f'{root}/derivatives/{sub}/{ses}/{base}_{modality}_skew_scaled.nii.gz'
                skewNii = nb.load(skewFile)
                skewData = skewNii.get_fdata()

                kurtFile = f'{root}/derivatives/{sub}/{ses}/{base}_{modality}_kurt_scaled.nii.gz'
                kurtNii = nb.load(kurtFile)
                kurtData = kurtNii.get_fdata()

                data1 = tSNRData[idxMask]
                data2 = meanData[idxMask]
                data3 = skewData[idxMask]
                data4 = kurtData[idxMask]

                for i, (value1, value2,value3,value4) in enumerate(zip(data1, data2,data3, data4)):
                    subList.append(sub)
                    modalityList.append(modality[:4])
                    tSNRList.append(value1)
                    voxelList.append(i)
                    meanList.append(value2)
                    runList.append(base)
                    focusList.append(focus)
                    skewList.append(value3)
                    kurtList.append(value4)


tSNRdata = pd.DataFrame({'subject':subList, 'modality':modalityList, 'tSNR': tSNRList, 'mean': meanList, 'voxel': voxelList, 'run':runList, 'focus':focusList, 'kurtosis':kurtList, 'skew':skewList})




v1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:blue'}
s1Palette = {
    'BOLD': 'tab:orange',
    'VASO': 'tab:green'}

palettes = [v1Palette,s1Palette]

for measure in ['mean','tSNR','skew','kurtosis']:


    for focus,palette in zip(['v1','s1'], palettes):
        tmp = tSNRdata.loc[(tSNRdata['focus']==focus)&(tSNRdata['run'].str.contains('run-001'))&(tSNRdata['run'].str.contains('block'))]

        fig, ax = plt.subplots()
        sns.kdeplot(data = tmp ,x=measure,hue='modality',linewidth=2,palette=palette)
        plt.title(f'{focus} ROI {measure}',fontsize=24)
        # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.ylabel('voxel count',fontsize=24)
        plt.yticks([])

        if measure == 'tSNR':
            ticks = np.arange(0,50,10)
            plt.xticks(ticks, fontsize=14)

        plt.xlabel(f'{measure}',fontsize=20)

        #legend hack
        old_legend = ax.legend_
        handles = old_legend.legendHandles
        labels = [t.get_text() for t in old_legend.get_texts()]
        title = old_legend.get_title().get_text()
        ax.legend(handles, labels, loc='upper right', title='', fontsize=18)
        plt.savefig(f'../results/group_{focus}_{measure}.png',bbox_inches='tight')

        plt.show()




for focus,palette in zip(['v1','s1'], palettes):


    tmp = tSNRdata.loc[(tSNRdata['focus']==focus)&(tSNRdata['run'].str.contains('run-001'))&(tSNRdata['run'].str.contains('block'))]

    fig, ax = plt.subplots()
    sns.kdeplot(data = tmp ,x='tSNR',hue='modality',linewidth=2,palette=palette)
    plt.title(f'{focus} ROI tSNR',fontsize=24)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.ylabel('voxel count',fontsize=24)
    plt.yticks([])
    ticks = np.arange(0,50,10)

    plt.xticks(ticks, fontsize=14)
    plt.xlabel('tSNR',fontsize=20)

    #legend hack
    old_legend = ax.legend_
    handles = old_legend.legendHandles
    labels = [t.get_text() for t in old_legend.get_texts()]
    title = old_legend.get_title().get_text()
    ax.legend(handles, labels, loc='upper right', title='', fontsize=18)
    plt.savefig(f'../results/group_{focus}_tSNR.png',bbox_inches='tight')

    plt.show()




# Get voxel numbers per ROI

subList = []
voxelCountList = []
roiList = []

for sub in subs:
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
        print(ses)
        for focus in ['v1', 's1']:
            try:
                if sub == 'sub-07':
                    maskFile = f'{root}/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_3layers_blockStim_layers_equidist.nii'

                else:
                    maskFile = f'{root}/derivatives/{sub}/{ses}/{sub}_masks/{sub}_{focus}_3layers_layers_equidist.nii'

                maskNii = nb.load(maskFile)
                maskData = maskNii.get_fdata()
                idxMask = maskData > 0

            except:
                print(f'no mask found for {focus}')
                continue

            subList.append(f'{sub} {ses}')
            voxelCountList.append(np.sum(idxMask))
            roiList.append(focus)




voxelCountData = pd.DataFrame({'subject':subList, 'focus':roiList, 'voxelCount':voxelCountList})


fig, ax = plt.subplots()
ax = sns.boxplot(y="voxelCount", x="focus",
                 data=voxelCountData)

ax.set_ylabel('voxel count',fontsize=24)
ax.set_xlabel('ROI',fontsize=24)
ax.tick_params(axis='both', labelsize=18)

plt.savefig(f'../results/roi_sizes.png',bbox_inches='tight')
plt.show()
