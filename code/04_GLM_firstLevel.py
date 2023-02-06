import os
import subprocess
import glob
import nibabel as nb
import re
import numpy as np
import time

FSFDIR = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti/derivativesTestTest/designFiles'
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
DERIVATIVES = f'{ROOT}/derivativesTestTest'
def findTR(logfile):
    with open(logfile) as f:
        f = f.readlines()

    triggerTimes = []
    for line in f[1:]:
        if re.findall("Keypress: 5",line):
            triggerTimes.append(float(re.findall("\d+\.\d+", line)[0]))

    triggerTimes[0] = 0

    triggersSubtracted = []
    for n in range(len(triggerTimes)-1):
        triggersSubtracted.append(float(triggerTimes[n+1])-float(triggerTimes[n]))

    meanFirstTriggerDur = np.mean(triggersSubtracted[::2])
    meanSecondTriggerDur = np.mean(triggersSubtracted[1::2])

    # find mean trigger-time
    meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2
    return meanTriggerDur

for sub in ['sub-13']:
    # for ses in ['ses-001','ses-002']:
    for ses in ['ses-001']:
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

        runTypes = []
        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-12].split('-')[-1]
            if base not in runTypes:
                runTypes.append(base)

        for runType in runTypes:
            print(f'Processing {runType}')

            runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-00*_cbv.nii.gz'))

            for i, run in enumerate(runs, start = 1):

                print(f'Processing {sub}_{ses}_task-{runType}_run-{i:03d}')

                # tr = findTR(f'{DERIVATIVES}/{sub}/{ses}/func/events/{sub}_{ses}_task-{runType}_run-{i:03d}.log')

                for modality in ['BOLD', 'VASO']:
                # for modality in ['VASO']:

                    actualData = f'{DERIVATIVES}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-{i:03d}_{modality}.nii.gz'

                    runData = nb.load(actualData)
                    tr = str(runData.header.get_zooms()[-1])

                    runData = runData.get_fdata()
                    nrVolumes = str(runData.shape[-1])
                    print(nrVolumes)

                    replacements = {'SUBID':f'{sub}','SESID':ses, 'RUNID': f'run-{i:03d}', 'ROOT':DERIVATIVES, 'NRVOLS': nrVolumes, 'TRVAL':tr, 'MODALITY':modality}

                    with open(f"{FSFDIR}/{runType}Template.fsf") as infile:
                        with open(f"{FSFDIR}/{sub}_{ses}_task-{runType}_run-{i:03d}_{modality}.fsf", 'w') as outfile:
                            for line in infile:
                                for src, target in replacements.items():
                                    line = line.replace(src, target)
                                outfile.write(line)


for sub in ['sub-13']:
        runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-001/func/{sub}_ses-00*_task-*run-00*_cbv.nii.gz'))
        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            for modality in ['BOLD', 'VASO']:
                file = f'{DERIVATIVES}/{sub}/ses-001/{base}_{modality}.feat/stats/zstat1.nii.gz'
                if not os.path.exists(file):
                    print(f'Processing run {base}_{modality}')
                    subprocess.run(f'feat {FSFDIR}/{base}_{modality}.fsf &', shell=True)
            time.sleep(60*20)





# for FIR analysis
for sub in ['sub-13']:
# for sub in ['sub-06']:

    ses='ses-001'
    runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-event*run-00*_cbv.nii.gz'))

    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/ses-001'

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')
        tr = findTR(f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log')

        for modality in ['BOLD', 'VASO']:
        # for modality in ['VASO']:

            actualData = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{base}_{modality}.nii.gz'

            runData = nb.load(actualData)
            tr = str(runData.header.get_zooms()[-1])

            runData = runData.get_fdata()
            nrVolumes = str(runData.shape[-1])
            print(nrVolumes)

            for focus in ['s1','v1']:

                replacements = {'SUBID':f'{sub}', 'BASE':base, 'FOCUS':focus, 'NRVOLS': nrVolumes, 'SESID':ses, 'MODALITY':modality}



                with open(f"{FSFDIR}/templateDesign_FIR.fsf") as infile:
                    with open(f"{FSFDIR}/{base}_{focus}_{modality}_FIR.fsf", 'w') as outfile:
                        for line in infile:
                            for src, target in replacements.items():
                                line = line.replace(src, target)
                            outfile.write(line)

for sub in ['sub-13']:

    ses='ses-001'
    runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-event*run-00*_cbv.nii.gz'))

    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/ses-001'

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')
        if 'Random' in base:
            type='_Random'
        else:
            type=''

        tr = findTR(f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log')

        for modality in ['BOLD', 'VASO']:
        # for modality in ['VASO']:

            actualData = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{base}_{modality}.nii.gz'

            runData = nb.load(actualData)
            tr = str(runData.header.get_zooms()[-1])

            runData = runData.get_fdata()
            nrVolumes = str(runData.shape[-1])
            print(nrVolumes)

            for focus in ['s1','v1']:

                replacements = {'SUBID':f'{sub}', 'BASE':base, 'FOCUS':focus, 'NRVOLS': nrVolumes, 'SESID':ses, 'MODALITY':modality}



                with open(f"{FSFDIR}/templateDesign_FIR{type}.fsf") as infile:
                    with open(f"{FSFDIR}/{base}_{focus}_{modality}_FIR.fsf", 'w') as outfile:
                        for line in infile:
                            for src, target in replacements.items():
                                line = line.replace(src, target)
                            outfile.write(line)

# short vs long ITI
subs = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']
subs = ['sub-09','sub-11']
for sub in subs:
# for sub in ['sub-06']:

    ses='ses-002'
    runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-event*run-00*_cbv.nii.gz'))

    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/ses-001'

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')
        if 'Random' in base:
            type='_Random'
        else:
            type=''

        tr = findTR(f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log')

        for modality in ['BOLD', 'VASO']:
        # for modality in ['VASO']:

            actualData = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{base}_{modality}.nii.gz'

            runData = nb.load(actualData)
            tr = str(runData.header.get_zooms()[-1])

            runData = runData.get_fdata()
            nrVolumes = str(runData.shape[-1])
            print(nrVolumes)

            for focus in ['v1']:

                replacements = {'SUBID':f'{sub}', 'BASE':base, 'FOCUS':focus, 'NRVOLS': nrVolumes, 'SESID':ses, 'MODALITY':modality}



                with open(f"{FSFDIR}/templateDesign_FIR{type}_longVsShortITI.fsf") as infile:
                    with open(f"{FSFDIR}/{base}_{focus}_{modality}_FIR_longVsShortITI.fsf", 'w') as outfile:
                        for line in infile:
                            for src, target in replacements.items():
                                line = line.replace(src, target)
                            outfile.write(line)