import numpy as np
import nibabel as nb
import glob
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy import stats
from pingouin import mwu


# compare tSNR for long vs short TR
subs = ['sub-09','sub-11']

modalities = ['BOLD', 'VASO']

subList = []
modalityList = []
tSNRList = []
meanList = []
voxelList = []
runList = []
focusList = []
TRlist = []


for sub in subs:
    print(sub)
    for type in ['blockStim','blockStimLongTR']:
        runs = sorted(glob.glob(f'{root}/{sub}/ses-001/func/{sub}_ses-00*_task-{type}_run*_cbv.nii.gz'))

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


                    data1 = tSNRData[idxMask]
                    data2 = meanData[idxMask]

                    for i, (value1, value2) in enumerate(zip(data1, data2)):
                        subList.append(sub)
                        modalityList.append(modality[:4])
                        tSNRList.append(value1)
                        voxelList.append(i)
                        meanList.append(value2)
                        runList.append(base)
                        focusList.append(focus)
                        TRlist.append(type)

tSNRdata = pd.DataFrame({'subject':subList, 'modality':modalityList, 'tSNR': tSNRList, 'mean': meanList, 'voxel': voxelList, 'run':runList, 'focus':focusList, 'TRlength':TRlist})




plt.style.use('dark_background')


g = sns.FacetGrid(tSNRdata, col="focus",  row="modality",hue='TRlength')
g.map(sns.kdeplot, "tSNR",linewidth=2)

# adapt y-axis
g.axes[0,0].set_yticks([])
g.axes[1,0].set_yticks([])

# plt.title(r'\fontsize{30pt}{3em}\selectfont{}{Mean WRFv3.5 LHF\r}{\fontsize{18pt}{3em}\selectfont{}voxel count}')


g.axes[0,0].set_ylabel('BOLD\nvoxel count',fontsize=18)
g.axes[1,0].set_ylabel('VASO\nvoxel count',fontsize=18)

# adapt x-axis
g.axes[1,0].set_xlabel('tSNR',fontsize=18)
g.axes[1,1].set_xlabel('tSNR',fontsize=18)
g.axes[1,1].tick_params(axis='x', labelsize=18)
g.axes[1,0].tick_params(axis='x', labelsize=18)


g.axes[0,0].set_title('v1',fontsize=18)
g.axes[0,1].set_title('s1',fontsize=18)
g.axes[1,0].set_title('',fontsize=18)
g.axes[1,1].set_title('',fontsize=18)



g.add_legend(fontsize='x-large')

plt.savefig(f'../results/tSNR_longVsShort.jpg')



for modality in modalities:
    print(modality)
    for focus in ['v1', 's1']:
        print(focus)
        # long TR data
        tmp1 = tSNRdata.loc[(tSNRdata['focus']==focus)&(tSNRdata['modality']==modality)&(tSNRdata['modality']==modality)&(tSNRdata['run'].str.contains('Long'))]['tSNR']
        # short TR data
        tmp2 = tSNRdata.loc[(tSNRdata['focus']==focus)&(tSNRdata['modality']==modality)&(tSNRdata['modality']==modality)&(~tSNRdata['run'].str.contains('Long'))]['tSNR']

        results2 = mwu(tmp1, tmp2)

        if results2['p-val'][0]<=0.05:
            print(f'effect size: {results2["CLES"][0]}') # CLES = common language effect size
            print('sig')



# compare layer profles


subList = []
dataList = []
modalityList = []
layerList = []
stimTypeList = []
focusList = []
contrastList = []

root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'



### block profiles

for sub in ['sub-09','sub-11']:

    print(sub)

    blockRuns = sorted(glob.glob(f'{root}/{sub}/ses-001/func/{sub}_*_task-block*run-00*_cbv.nii.gz'))




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

            for task in ['blockStimLongTR','blockStim']:

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
zscores = pd.DataFrame({'subject': subList, 'data': dataList, 'modality': modalityList, 'layer':layerList, 'stimType':stimTypeList, 'contrast':contrastList,'focus':focusList})

zscores




plt.style.use('dark_background')


g = sns.FacetGrid(zscores, col="focus",  row="modality", hue='stimType',sharey=False,sharex=False)
g.map(sns.lineplot,'layer',"data",linewidth=2)

# adapt y-axis

g.axes[1,0].tick_params(axis='y', labelsize=18)
g.axes[0,0].tick_params(axis='y', labelsize=18)
g.axes[1,1].tick_params(axis='y', labelsize=18)
g.axes[0,1].tick_params(axis='y', labelsize=18)




g.axes[0,0].set_ylabel('BOLD\nz-scores',fontsize=18)
g.axes[1,0].set_ylabel('VASO\nz-scores',fontsize=18)

# adapt x-axis
g.axes[0,0].set_xticks([],fontsize=18)
g.axes[0,1].set_xticks([],fontsize=18)

g.axes[1,0].set_xticks([],fontsize=18)
g.axes[1,1].set_xticks([],fontsize=18)

g.axes[1,0].set_xlabel('WM                 CSF',fontsize=18)
g.axes[1,1].set_xlabel('WM                 CSF',fontsize=18)




g.axes[0,0].set_title('v1',fontsize=18)
g.axes[0,1].set_title('s1',fontsize=18)
g.axes[1,0].set_title('',fontsize=18)
g.axes[1,1].set_title('',fontsize=18)



g.add_legend(fontsize='x-large')
plt.savefig(f'../results/profiles_longVsShort.jpg',bbox_inches='tight')



                 
