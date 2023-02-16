"""

Prepare and run second level GLM in FSL FEAT.
Requires registration-workaround to have run.

Assumes template files for each type of run.

"""

import os
import subprocess
import glob
import time

# Set some folder names
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
DERIVATIVES = f'{ROOT}/derivativesTestTest'

subs = ['sub-08']

for sub in subs:
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

            for modality in ['BOLD', 'VASO']:

                    for modelType in ['conv', 'fir']:
                        if modelType == 'fir' and 'block' in runType:
                            continue

                        # Check if there are multiple runs of runType in session
                        runTypeRuns = glob.glob(
                            f'{DERIVATIVES}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-*_{modality}_{modelType}.feat')

                        nrRuns = len(runTypeRuns)

                        if nrRuns >= 2:
                            replacements = {'SUBID': f'{sub}',
                                            'SESID': ses,
                                            'ROOT': DERIVATIVES,
                                            'MODALITY': modality
                                            }

                            with open(f"{DERIVATIVES}/designFiles/{runType}SecondLevelTemplate{nrRuns}Inputs_{modelType}.fsf") as infile:
                                with open(f"{DERIVATIVES}/designFiles/{sub}_{ses}_task-{runType}_secondLevel_{modality}_{modelType}.fsf", 'w') as outfile:
                                    for line in infile:
                                        for src, target in replacements.items():
                                            line = line.replace(src, target)
                                        outfile.write(line)




# =====================================================================================================================
# Run GLMs
# =====================================================================================================================

for sub in subs:
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

            for modality in ['BOLD', 'VASO']:

                    for modelType in ['conv', 'fir']:
                        if modelType == 'fir' and 'block' in runType:
                            continue

                        # Check if there are multiple runs of runType in session
                        runTypeRuns = glob.glob(
                            f'{DERIVATIVES}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-*_{modality}_{modelType}.feat')

                        nrRuns = len(runTypeRuns)

                        if nrRuns >= 2:

                            # Check if the GLM for this run already ran
                            file = f'{DERIVATIVES}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_secondLevel_{modality}_{modelType}.gfeat/cope1.feat/stats/zstat1.nii.gz'

                            if os.path.exists(file):  # If yes, skip
                                print(f'Second level GLM for {sub} {ses} {runType} {modality} {modelType} already ran')

                            if not os.path.exists(file):  # If no, run
                                subprocess.run(f'feat {DERIVATIVES}/designFiles/{sub}_{ses}_task-{runType}_secondLevel_{modality}_{modelType}.fsf &', shell=True)

