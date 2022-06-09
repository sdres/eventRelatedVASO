import os
import subprocess
import glob
import nibabel as nb
import re
import numpy as np


fsfDir='/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/designFiles'
root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

for sub in ['sub-09']:
    runs = sorted(glob.glob(f'{root}/{sub}/ses-001/func/{sub}_ses-00*_task-*run-00*_cbv.nii.gz'))

    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/ses-001'

    # for stimType in ['blockStimVisOnly']:
    for stimType in ['blockStim', 'blockStimLongTR']:
    # for stimType in ['blockStim', 'eventStim']:
    # for stimType in ['eventStimRandom']:
        runs = sorted(glob.glob(f'{root}/{sub}/ses-001/func/{sub}_ses-00*_task-{stimType}_run-00*_cbv.nii.gz'))

        nrRuns = int(len(runs))
        ses = 'ses-001'

        for modality in ['BOLD', 'VASO']:

            replacements = {'SUBID':f'{sub}', 'SESID':ses, 'MODALITY': modality, 'STIMTYPE':stimType}

            with open(f"{fsfDir}/templateDesignSecondLevel{nrRuns}Inputs.fsf") as infile:
                with open(f"{fsfDir}/{sub}_{ses}_task-{stimType}_secondLevel_{modality}.fsf", 'w') as outfile:
                    for line in infile:
                        for src, target in replacements.items():
                            line = line.replace(src, target)
                        outfile.write(line)
