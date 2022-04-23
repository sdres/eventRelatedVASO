import os
import subprocess
import glob
import nibabel as nb
import re
import numpy as np
import time

fsfDir='/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/designFiles'
root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

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

for sub in ['sub-05']:
# for sub in ['sub-06']:
    # for ses in ['ses-001','ses-002']:
    for ses in ['ses-001']:
        runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

        outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}'

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')
            tr = findTR(f'{root}/derivatives/{sub}/{ses}/events/{base}.log')

            for modality in ['BOLD', 'VASO']:
            # for modality in ['VASO']:

                actualData = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{base}_{modality}.nii.gz'

                runData = nb.load(actualData)
                tr = str(runData.header.get_zooms()[-1])

                runData = runData.get_fdata()
                nrVolumes = str(runData.shape[-1])
                print(nrVolumes)


                replacements = {'SUBID':f'{sub}','SESID':ses, 'BASE':base, 'NRVOLS': nrVolumes, 'TRVAL':tr}



                with open(f"{fsfDir}/templateDesign{modality}.fsf") as infile:
                    with open(f"{fsfDir}/{base}_{modality}.fsf", 'w') as outfile:
                        for line in infile:
                            for src, target in replacements.items():
                                line = line.replace(src, target)
                            outfile.write(line)


for sub in ['sub-09']:
        runs = sorted(glob.glob(f'{root}/{sub}/ses-002/func/{sub}_ses-00*_task-*run-00*_cbv.nii.gz'))
        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            for modality in ['BOLD', 'VASO']:
                file = f'{root}/derivatives/{sub}/ses-002/{base}_{modality}.feat/stats/zstat1.nii.gz'
                if not os.path.exists(file):
                    print(f'Processing run {base}_{modality}')
                    subprocess.run(f'feat {fsfDir}/{base}_{modality}.fsf &',shell=True)
            time.sleep(60*20)





# for FIR analysis
for sub in ['sub-13']:
# for sub in ['sub-06']:

    ses='ses-001'
    runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-event*run-00*_cbv.nii.gz'))

    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/ses-001'

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')
        tr = findTR(f'{root}/derivatives/{sub}/{ses}/events/{base}.log')

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



                with open(f"{fsfDir}/templateDesign_FIR.fsf") as infile:
                    with open(f"{fsfDir}/{base}_{focus}_{modality}_FIR.fsf", 'w') as outfile:
                        for line in infile:
                            for src, target in replacements.items():
                                line = line.replace(src, target)
                            outfile.write(line)

for sub in ['sub-13']:
# for sub in ['sub-06']:

    ses='ses-001'
    runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-event*run-00*_cbv.nii.gz'))

    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/ses-001'

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')
        if 'Random' in base:
            type='_Random'
        else:
            type=''

        tr = findTR(f'{root}/derivatives/{sub}/{ses}/events/{base}.log')

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



                with open(f"{fsfDir}/templateDesign_FIR{type}.fsf") as infile:
                    with open(f"{fsfDir}/{base}_{focus}_{modality}_FIR.fsf", 'w') as outfile:
                        for line in infile:
                            for src, target in replacements.items():
                                line = line.replace(src, target)
                            outfile.write(line)


# for VESSEL FIR analysis
for sub in ['sub-14']:
# for sub in ['sub-06']:
    ses='ses-001'
    runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-event*run-00*_cbv.nii.gz'))

    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/ses-001'

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')
        tr = findTR(f'{root}/derivatives/{sub}/{ses}/events/{base}.log')

        for modality in ['BOLD', 'VASO']:
        # for modality in ['VASO']:

            actualData = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}/{base}_{modality}.nii.gz'

            runData = nb.load(actualData)
            tr = str(runData.header.get_zooms()[-1])

            runData = runData.get_fdata()
            nrVolumes = str(runData.shape[-1])
            print(nrVolumes)


            replacements = {'SUBID':f'{sub}', 'BASE':base, 'NRVOLS': nrVolumes, 'SESID':ses, 'MODALITY':modality}



            with open(f"{fsfDir}/templateDesign_vesselFIR.fsf") as infile:
                with open(f"{fsfDir}/{base}_{modality}_vesselFIR.fsf", 'w') as outfile:
                    for line in infile:
                        for src, target in replacements.items():
                            line = line.replace(src, target)
                        outfile.write(line)





fsfDir

fsfs = sorted(glob.glob(f'{fsfDir}/sub-13_ses-001_*v1*_FIR*'))


for file in fsfs:
    subprocess.run(f'feat {file}', shell=True)
