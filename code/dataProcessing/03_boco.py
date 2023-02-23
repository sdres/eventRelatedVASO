"""

Performing part of the VASO pipeline. Most crucially:
- Temporally upsampling data with factor of 2
- Fixing TR in header
- Prepending volume to nulled data to match timing
- BOLD-correction
- Misc data reduction
- QA

"""

import subprocess
import glob
import os
import nibabel as nb
import numpy as np
import re

# Define some directories
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'


def findTR(logfile):
    with open(logfile) as f:
        f = f.readlines()

    triggerTimes = []
    for line in f[1:]:
        if re.findall("Keypress: 5", line):
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


# SUBS = ['sub-09','sub-11']
SUBS = ['sub-09']

for sub in SUBS:
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
        for i in range(1, 3):  # We had a maximum of 2 sessions
            if f'ses-00{i}' in run:
                sessions.append(f'ses-00{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    # Loop over sessions
    for ses in sessions:
        print(f'Working on {ses}')

        # Set folder for session outputs
        outFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'

        # Look for individual runs within session (containing both nulled and notnulled images)
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')

            for start, modality in enumerate(['notnulled', 'nulled']):
                print(f'Starting with {modality}, indexed with {start}')

                command = f'3dUpsample '
                command += f'-overwrite '
                command += f'-datum short '
                command += f'-prefix {outFolder}/{base}_{modality}_intemp.nii '
                command += f'-n 2 '
                command += f'-input {outFolder}/{base}_{modality}_moco.nii'

                subprocess.call(command, shell=True)

                TR = findTR(f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log')

                # fix TR in header
                subprocess.call(
                    f'3drefit -TR {TR} '
                    + f'{outFolder}'
                    + f'/{base}_{modality}_intemp.nii',
                    shell=True
                    )

                # =====================================================================
                # Duplicate first nulled timepoint to match timing between cbv and bold
                # =====================================================================

                if modality == 'nulled':
                    nii = nb.load(
                        f'{outFolder}'
                        + f'/{base}_{modality}_intemp.nii'
                        )

                    # Load data
                    data = nii.get_fdata()  # Get data
                    header = nii.header  # Get header
                    affine = nii.affine  # Get affine

                    # Make new array
                    newData = np.zeros(data.shape)

                    for i in range(data.shape[-1]):
                        if i == 0:
                            newData[..., i] = data[..., i]
                        else:
                            newData[..., i] = data[..., i-1]

                    # Save data
                    img = nb.Nifti1Image(newData, header=header, affine=affine)
                    nb.save(img,
                            f'{outFolder}'
                            + f'/{base}_{modality}_intemp.nii'
                            )

            # ==========================================================================
            # BOLD-correction
            # ==========================================================================
            print(f'Starting BOCO')

            nulledFile = f'{outFolder}/{base}_nulled_intemp.nii'
            notnulledFile = f'{outFolder}/{base}_notnulled_intemp.nii'

            # Load data
            nii1 = nb.load(nulledFile).get_fdata()  # Load cbv data
            nii2 = nb.load(notnulledFile).get_fdata()  # Load BOLD data

            # Find timeseries with fewer volumes
            if nii1.shape[-1] < nii2.shape[-1]:
                maxTP = nii1.shape[-1]
            elif nii1.shape[-1] > nii2.shape[-1]:
                maxTP = nii2.shape[-1]
            else:
                maxTP = nii1.shape[-1]-1

            header = nb.load(nulledFile).header  # Get header
            affine = nb.load(nulledFile).affine  # Get affine

            # Divide VASO by BOLD for actual BOCO
            new = np.divide(nii1[..., :maxTP], nii2[..., :maxTP])

            # Clip range to -1.5 and 1.5. Values should be between 0 and 1 anyway.
            new[new > 1.5] = 1.5
            new[new < -1.5] = -1.5

            # Save BOLD-corrected VASO image
            img = nb.Nifti1Image(new, header=header, affine=affine)

            nb.save(
                img, f'{outFolder}'
                + f'/{base}_VASO_LN.nii'
                )

            # Multiply VASO by 100 because GLM was faulty if data was scaled between 0 and 1
            command = 'fslmaths '
            command += f'{outFolder}/{base}_VASO_LN.nii '
            command += '-mul 100 '
            command += f'{outFolder}/{base}_VASO.nii.gz '
            command += '-odt short'
            subprocess.run(command, shell=True)

            # Save BOLD with new name and short datatype
            command = 'fslmaths '
            command += f'{outFolder}/{base}_notnulled_intemp.nii '
            command += '-mul 1 '
            command += f'{outFolder}/{base}_BOLD.nii.gz '
            command += '-odt short'
            subprocess.run(command, shell=True)

            print('Calculating quality measures')
            for modality in ['BOLD', 'VASO']:
                subprocess.run(f'LN_SKEW -input {outFolder}/{base}_{modality}.nii.gz', shell=True)
