"""

Prepare and run first level GLM in FSL FEAT.
Assumes template files for each type of run.

"""

import os
import subprocess
import glob
import nibabel as nb
import time

# Set some folder names
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
DERIVATIVES = f'{ROOT}/derivativesTestTest'

for sub in ['sub-08']:
    # Find all runs of participant
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-*_task-*run-00*_cbv.nii.gz'))

    # Find sessions
    sessions = []
    for run in runs:
        for i in range(1, 3):
            tmp = f'ses-{i:03d}'
            sessions.append(tmp)
    sessions = set(sessions)

    # Loop over sessions
    for ses in sessions:
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

        runTypes = []
        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-12].split('-')[-1]
            if base not in runTypes:
                runTypes.append(base)

        for runType in runTypes:
            print(f'Processing {runType}')
            if runType == 'blockStimRandom':
                continue
            runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-00*_cbv.nii.gz'))

            for i, run in enumerate(runs, start=1):
                print(f'Processing {sub}_{ses}_task-{runType}_run-{i:03d}')

                for modality in ['BOLD', 'VASO']:

                    actualData = f'{DERIVATIVES}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-{i:03d}_{modality}.nii.gz'

                    runData = nb.load(actualData)
                    tr = str(runData.header.get_zooms()[-1])  # Get number of volumes from header
                    nrVolumes = str(runData.header['dim'][4])

                    replacements = {'SUBID': f'{sub}',
                                    'SESID': ses,
                                    'RUNID': f'run-{i:03d}',
                                    'ROOT': DERIVATIVES,
                                    'NRVOLS': nrVolumes,
                                    'TRVAL': tr,
                                    'MODALITY': modality
                                    }

                    for modelType in ['conv', 'fir']:
                        if modelType == 'fir' and 'block' in runType:
                            continue

                        with open(f"{DERIVATIVES}/designFiles/{runType}Template_{modelType}{modality}.fsf") as infile:
                            with open(f"{DERIVATIVES}/designFiles/{sub}_{ses}_task-{runType}_run-{i:03d}_{modality}_{modelType}.fsf", 'w') as outfile:
                                for line in infile:
                                    for src, target in replacements.items():
                                        line = line.replace(src, target)
                                    outfile.write(line)

# =====================================================================================================================
# Run GLMs
# =====================================================================================================================

executed = 0  # Counter for parallel processes
for sub in ['sub-08']:
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-00*/func/{sub}_ses-00*_task-*run-00*_cbv.nii.gz'))
    nrRuns = len(runs)
    print(f'Found {nrRuns} runs')

    for run in runs:
        # Set basename of run
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]

        for modality in ['BOLD', 'VASO']:  # Loop over modalities

            for modelType in ['conv', 'fir']:

                if modelType == 'fir' and 'block' in base:
                    print('Skipping fir for block-wise stimulation')
                    continue

                # Check if the GLM for this run already ran
                file = f'{DERIVATIVES}/{sub}/ses-001/func/{base}_{modality}_{modelType}.feat/stats/zstat1.nii.gz'

                if os.path.exists(file):  # If yes, skip
                    print(f'GLM for {base}_{modality} already ran')

                if not os.path.exists(file):  # If no, run
                    print(f'Processing run {base}_{modality} {modelType}')
                    subprocess.run(f'feat {DERIVATIVES}/designFiles/{base}_{modality}_{modelType}.fsf &', shell=True)
                    executed += 1  # Count parallel processes

                # Wait 30 minutes before starting to process next set of runs if 2 runs are being processed
                if executed >= 2:
                    time.sleep(60*30)
                    executed = 0  # Reset counter
