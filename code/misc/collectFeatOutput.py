"""Copying and renaming FEAT output to statsMap folder"""

import subprocess
import glob
import re
import os
import nibabel as nb


ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08']
for sub in subs:
    print('\n\n')
    print(sub)
    # Find all runs of participant
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-*_task-*run-00*_cbv.nii.gz'))

    # Find sessions
    sessions = []
    for run in runs:
        for i in range(1, 3):
            tmp = f'ses-{i:03d}'
            if f'ses-{i:03d}' in run:
                sessions.append(tmp)
    sessions = set(sessions)
    print(f'Found sessions: {sessions}')

    # Loop over sessions
    for ses in sessions:
        print(ses)
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

        # Set folder for session outputs
        sesFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'
        statFolder = f'{sesFolder}/statMaps'

        # Create folder if it does not exist
        if not os.path.exists(statFolder):
            os.makedirs(statFolder)
            print("statMap directory is created")

        runTypes = []
        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-12].split('-')[-1]
            if base not in runTypes:
                runTypes.append(base)

        for runType in runTypes:
            print(f'Processing {runType}')
            if runType == 'blockStimRandom':
                continue
            for modality in ['VASO', 'BOLD']:
                print(modality)

                for modelType in ['fir', 'conv']:
                    print(f'Model: {modelType}')
                    if 'blockStim' in runType and modelType == 'fir':
                        print(f'Skipping fir for blockStim')
                        continue

                    if 'Random' not in runType and 'VisOnly' not in runType:

                        if modelType == 'conv':
                            try:
                                secondLevel = sorted(glob.glob(f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_secondLevel_{modality}_{modelType}.gfeat'))
                                file = f'{secondLevel[0]}/cope1.feat/stats/zstat1.nii.gz'
                                tmp = nb.load(file)
                                command = f'cp {file} '
                                command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-visuotactile_{modality}_model-{modelType}_zstat.nii.gz'
                                # print(command)
                                subprocess.run(command, shell=True)

                            except:
                                firstLevel = sorted(glob.glob(f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_run-001_{modality}_{modelType}.feat'))
                                file = f'{firstLevel[0]}/stats/zstat1.nii.gz'
                                tmp = nb.load(file)
                                command = f'cp {file} '
                                command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-visuotactile_{modality}_model-{modelType}_zstat.nii.gz'
                                subprocess.run(command, shell=True)
                                # print(command)

                        if modelType == 'fir':
                            secondLevel = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_secondLevel_{modality}_{modelType}.gfeat'
                            files = sorted(glob.glob(f'{secondLevel}/cope*.feat/stats/cope1.nii.gz'))
                            tmp = nb.load(files[0])

                            for file in files:
                                cope = file.split('/')[-3]
                                number = int(re.findall(r'\d+', cope)[0])

                                command = f'cp {file} '
                                command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-visuotactile_modality-{modality}_model-{modelType}_cope-{number:02d}.nii.gz'
                                subprocess.run(command, shell=True)

                    if 'Random' in runType:
                        for i, contrast in enumerate(['visual', 'visuotactile'], start=1):

                            if modelType == 'conv':
                                try:
                                    secondLevel = sorted(glob.glob(f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_secondLevel_{modality}_{modelType}.gfeat'))
                                    file = f'{secondLevel[0]}/cope{i}.feat/stats/zstat1.nii.gz'
                                    tmp = nb.load(file)
                                    command = f'cp {file} '
                                    command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-{contrast}_modality-{modality}_model-{modelType}_zstat.nii.gz'
                                    subprocess.run(command, shell=True)
                                except:
                                    firstLevel = sorted(glob.glob(f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_run-001_{modality}_{modelType}.feat'))
                                    file = f'{firstLevel[0]}/stats/zstat{i}.nii.gz'
                                    tmp = nb.load(file)
                                    command = f'cp {file} '
                                    command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-{contrast}_modality-{modality}_model-{modelType}_zstat.nii.gz'
                                    subprocess.run(command, shell=True)

                        if modelType == 'fir':
                            secondLevel = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_secondLevel_{modality}_{modelType}.gfeat'
                            files = sorted(glob.glob(f'{secondLevel}/cope*.feat/stats/cope1.nii.gz'))
                            tmp = nb.load(files[0])

                            for file in files:
                                cope = file.split('/')[-3]
                                number = int(re.findall(r'\d+', cope)[0])
                                # print(f'Cope number: {number}')
                                if number <=10:
                                    command = f'cp {file} '
                                    command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-visual_modality-{modality}_model-{modelType}_cope-{number:02d}.nii.gz'
                                    subprocess.run(command, shell=True)
                                    # print(f'saving as visual {number:02d}')

                                if number >=11:
                                    command = f'cp {file} '
                                    command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-visuotactile_modality-{modality}_model-{modelType}_cope-{number-10:02d}.nii.gz'
                                    subprocess.run(command, shell=True)
                                    # print(f'saving as visuotactile {number-10:02d}')

                    if 'VisOnly' in runType:
                        if modelType == 'conv':
                            try:
                                secondLevel = sorted(glob.glob(f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_secondLevel_{modality}_{modelType}.gfeat'))
                                file = f'{secondLevel[0]}/cope1.feat/stats/zstat1.nii.gz'
                                tmp = nb.load(file)
                                command = f'cp {file} '
                                command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-visual_modality-{modality}_model-{modelType}_zstat.nii.gz'
                                subprocess.run(command, shell=True)
                            except:
                                firstLevel = sorted(glob.glob(f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_run-001_{modality}_{modelType}.feat'))
                                file = f'{firstLevel[0]}/stats/zstat1.nii.gz'
                                tmp = nb.load(file)
                                command = f'cp {file} '
                                command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-visual_modality-{modality}_model-{modelType}_zstat.nii.gz'
                                subprocess.run(command, shell=True)

                        if modelType == 'fir':
                            secondLevel = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/*{runType}_secondLevel_{modality}_{modelType}.gfeat'
                            files = sorted(glob.glob(f'{secondLevel}/cope*.feat/stats/cope1.nii.gz'))
                            tmp = nb.load(files[0])

                            for file in files:
                                cope = file.split('/')[-3]
                                number = int(re.findall(r'\d+', cope)[0])

                                command = f'cp {file} '
                                command += f'{statFolder}/{sub}_{ses}_task-{runType}_contrast-visual_modality-{modality}_model-{modelType}_cope-{number:02d}.nii.gz'
                                subprocess.run(command, shell=True)
