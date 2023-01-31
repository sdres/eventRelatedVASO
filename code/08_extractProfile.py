'''

Extracting profiles for block and event wise stimulation

'''

import numpy as np
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.ticker import FormatStrFormatter
import matplotlib.gridspec as gridspec

# Disable pandas warining
pd.options.mode.chained_assignment = None

# Set folder where data is found
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

# Set participants to work on
subs = ['sub-05','sub-06', 'sub-07','sub-08', 'sub-09','sub-11', 'sub-12','sub-13', 'sub-14']
# Done
subs = ['sub-06','sub-08']
# To do
subs = ['sub-07']

# Set modalities to work on
modalities = ['BOLD', 'VASO']
# modalities = ['BOLD']

# # Define layer names
# layerNames = {1:'deep',2:'middle',3:'superficial'}

# Initialize lists
subList = []  # For participants
modalityList = []  # For modalities
layerList = []  # For layers
timepointList = []  # For volumes
dataList = []  # For extracted values
focusList = []  # For v1/s1 focus
contrastList = []
runTypeList = []

for sub in subs:
    print(f'Working on {sub}')

    # Set folder for subject
    subFolder = f'{ROOT}/derivativesTestTest/{sub}'

    # =========================================================================
    # Look for sessions
    # Collectall runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/*/func/{sub}_*_task-*run-00*_cbv.nii.gz'))

    # Initialte list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1,3):  # We had a maximum of 2 sessions
            if f'ses-00{i}' in run:
                sessions.append(f'ses-00{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    # Loop over sessions
    for ses in sessions:
        print(f'Working on {ses}')

        # Set folder for session outputs
        sesFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'
        statFolder = f'{sesFolder}/statMaps'
        # Set folder where layer-ROIs are stored
        roiFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/rois'

        runTypes = []
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task*_run-00*_cbv.nii.gz'))

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-12].split('-')[-1]
            if base not in runTypes:
                runTypes.append(base)


        for runType in runTypes:
            print(f'Processing {runType}')

            contrastTypes = []

            if not ('Random' in runType) and not ('VisOnly' in runType):
                # print('only contrast is visiotactile')
                contrastTypes.append('visiotactile')

            if 'Random' in runType:
                # print('both visual and visiotactile')
                contrastTypes.append('visiotactile')
                contrastTypes.append('visual')

            if 'VisOnly' in runType:
                print('only contrast is visual')
                contrastTypes.append('visual')


            for modality in ['VASO', 'BOLD']:
                print(f'Processing {modality}')


                for contrastType in contrastTypes:


                    # for focus in ['v1', 's1']:
                    for focus in ['v1']:

                        # Load layer file in nibabel
                        layerFile = f'{roiFolder}/{sub}_rim-{focus}_11layers_layers_equivol.nii'
                        # layerFile = f'/Users/sebastiandresbach/data/eventRelatedVASO/Nifti/derivatives/sub-06/ses-001/sub-06_masks/sub-06_s1_11layers_layers_equidist.nii'
                        layerNii = nb.load(layerFile)
                        layerData = layerNii.get_fdata()  # Load as array
                        layers = np.unique(layerData)[1:]  # Look for number of layers


                        map = sorted(glob.glob(f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/upsample/{sub}_{runType}_{modality}_{contrastType}_ups5x.nii'))[0]


                        dataNii = nb.load(map)
                        data = dataNii.get_fdata()


                        for layer in layers:

                            idxLayer = layerData == layer

                            mean = np.mean(data[idxLayer])

                            subList.append(sub)
                            modalityList.append(modality)
                            layerList.append(layer)
                            focusList.append(focus)
                            contrastList.append(contrastType)
                            runTypeList.append(runType)
                            if modality =='BOLD':
                                dataList.append(mean)
                            if modality =='VASO':
                                dataList.append(mean)


zscoreData = pd.DataFrame({'subject':subList, 'layer':layerList, 'modality':modalityList, 'data':dataList, 'contrast':contrastList, 'focus':focusList, 'runType':runTypeList})


for modality in modalities:

    tmp = zscoreData.loc[(zscoreData['modality']==modality)&(zscoreData['contrast']=='visiotactile')]
    plt.figure()
    sns.lineplot(data=tmp, x='layer',y='data',hue='runType')
    plt.show()
