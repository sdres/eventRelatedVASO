import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
import numpy as np
root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'
plt.style.use('dark_background')
v1Palette = {
    'notnulled': 'tab:orange',
    'nulled': 'tab:blue'}
for sub in ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-10', 'sub-11','sub-12', 'sub-13', 'sub-14']:
# for sub in ['sub-05']:

    runs = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-*run-00*_cbv.nii.gz'))

    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/ses-001'


    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')

        if 'ses-001' in run:
            ses = 'ses-001'
        if 'ses-002' in run:
            ses = 'ses-002'
        outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}'

        rotTrans = []
        newName = []

        motionData = pd.read_csv(f'{outFolder}/motionParameters/{base}_motionParameters.csv')
        for i, row in motionData.iterrows():
            if 'R' in row['Motion_Name']:
                rotTrans.append('rotation')
                newName.append(row['Motion_Name'][1:])
            if 'T' in row['Motion_Name']:
                rotTrans.append('translation')
                newName.append(row['Motion_Name'][1:])

        motionData['type'] = rotTrans
        motionData['Motion_Name'] = newName

        motionData_rot_nulled = motionData.loc[(motionData['type'].str.contains("rotation") == 1)&(motionData['modality']=='nulled')].dropna()
        motionData_trans_nulled = motionData.loc[(motionData['type'].str.contains("translation") == 1)&(motionData['modality']=='nulled')].dropna()

        motionData_rot_notnulled = motionData.loc[(motionData['type'].str.contains("rotation") == 1)&(motionData['modality']=='notnulled')].dropna()
        motionData_trans_notnulled = motionData.loc[(motionData['type'].str.contains("translation") == 1)&(motionData['modality']=='notnulled')].dropna()


        fig, axes = plt.subplots(1, 2,sharex=True,figsize=(30,6))
        plt.suptitle(f'{base} Motion Summary', fontsize=24)

        width = 2

        sns.lineplot(ax=axes[0], x='Time/TR',y='Motion',data=motionData_trans_nulled, hue='Motion_Name', palette = 'Set1',linewidth = width,legend=False)
        sns.lineplot(ax=axes[0], x='Time/TR',y='Motion',data=motionData_trans_notnulled, hue='Motion_Name', palette = 'Set2',linewidth = width,legend=False)


        axes[0].set_ylabel("Translation [mm]", fontsize=24)
        # axes[0].legend(fontsize=20)

        axes[0].set_xlabel("Volume", fontsize=24)

        sns.lineplot(ax=axes[1], x='Time/TR',y='Motion',data=motionData_rot_nulled,hue='Motion_Name', palette = 'Set1',linewidth = width)
        sns.lineplot(ax=axes[1], x='Time/TR',y='Motion',data=motionData_rot_notnulled,hue='Motion_Name', palette = 'Set2',linewidth = width)


        # mylabels = ['X VASO','Y VASO','Z VASO','X BOLD','Y BOLD','Z BOLD']
        axes[1].set_xlabel("Volume", fontsize=24)
        axes[1].set_ylabel("Rotation [radians]", fontsize=24)
        # axes[1].legend(fontsize=20)
        # axes[1].legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5),labels=mylabels)
        axes[1].legend(fontsize=20, loc='center left', bbox_to_anchor=(1, 0.5))
        axes[0].tick_params(axis='both', labelsize=20)
        axes[1].tick_params(axis='both', labelsize=20)

        mylabels = ['X VASO','Y VASO','Z VASO','X BOLD','Y BOLD','Z BOLD']

        # axes[1].legend(labels=mylabels)

        plt.savefig(f'../results/motionParameters/{base}_motion.jpg', bbox_inches = 'tight', pad_inches = 0)
        plt.show()






        sub_FD = []
        timepoints = []
        subjects=[]
        mods = []
        runList = []


        for modality in ['nulled', 'notnulled']:

            TX = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("X") == 1)&(motionData['modality']==modality)&(motionData['type']=='translation')].tolist()
            TY = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("Y") == 1)&(motionData['modality']==modality)&(motionData['type']=='translation')].tolist()
            TZ = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("Z") == 1)&(motionData['modality']==modality)&(motionData['type']=='translation')].tolist()

            RX = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("X") == 1)&(motionData['modality']==modality)&(motionData['type']=='rotation')].tolist()
            RY = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("Y") == 1)&(motionData['modality']==modality)&(motionData['type']=='rotation')].tolist()
            RZ = motionData['Motion'].loc[(motionData['Motion_Name'].str.contains("Z") == 1)&(motionData['modality']==modality)&(motionData['type']=='rotation')].tolist()

            for n in range(len(TX)-4):
                FD_trial = abs(TX[n]-TX[n+1])+abs(TY[n]-TY[n+1])+abs(TZ[n]-TZ[n+1])+abs((50*RX[n])-(50*RX[n+1]))+abs((50*RY[n])-(50*RY[n+1]))+abs((50*RZ[n])-(50*RZ[n+1]))
                sub_FD.append(FD_trial)
                timepoints.append(n)
                subjects.append(sub)
                mods.append(modality)
                # runList.append(base)


        FDs = pd.DataFrame({'subject':subjects, 'volume':timepoints, 'FD':sub_FD, 'modality': mods})
        FDs.to_csv(f'{root}/derivatives/{sub}/{ses}/motionParameters/{base}_FDs.csv', index=False)

        fig = plt.figure(figsize=(20,5))
        sns.lineplot(data=FDs, x='volume', y='FD',hue='modality',linewidth = width, palette=v1Palette)

        if np.max(FDs['FD']) < 0.88:
            plt.ylim(0,1)


        plt.axhline(0.88, color='gray', linestyle='--')
        plt.ylabel('FD [mm]', fontsize=24)
        plt.xlabel('Volume', fontsize=24)
        plt.title(f"{base}", fontsize=24, pad=20)
        plt.legend(fontsize=20)


        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.savefig(f'../results/motionParameters/{base}_FDs.png', bbox_inches = 'tight', pad_inches = 0)
        plt.show()
