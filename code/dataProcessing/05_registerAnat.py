'''

Registering anatomical to functional images

'''

import glob
import os
import subprocess
import nibabel as nb

SUBS = ['sub-14']
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

# Define data dir
DATADIR = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti/derivativesTestTest'


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
        for i in range(1,3):  # We had a maximum of 2 sessions
            if f'ses-00{i}' in run:
                sessions.append(f'ses-00{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    # Loop over sessions
    for ses in sessions:
        print(f'Working on {ses}')

        # Defining folders
        funcDir = f'{DATADIR}/{sub}/{ses}/func'  # Location of functional data
        anatDir = f'{DATADIR}/{sub}/ses-001/anat'  # Location of anatomical data

        regFolder = f'{anatDir}/registrationFiles'  # Folder where output will be saved
        # Make output folder if it does not exist already
        if not os.path.exists(regFolder):
            os.makedirs(regFolder)

        # =========================================================================
        # Registration
        # =========================================================================

        moving = glob.glob(f'{anatDir}/{sub}_brain.nii*')[0]
        fixed = f'{funcDir}/{sub}_{ses}_T1w.nii'

        # Set up ants command
        command = 'antsRegistration '
        command += f'--verbose 1 '
        command += f'--dimensionality 3 '
        command += f'--float 0 '
        command += f'--collapse-output-transforms 1 '
        command += f'--interpolation BSpline[5] '
        command += f'--output [{regFolder}/registered1_,{regFolder}/registered1_Warped.nii,1] '
        command += f'--use-histogram-matching 0 '
        command += f'--winsorize-image-intensities [0.005,0.995] '
        command += f'--initial-moving-transform {anatDir}/objective_matrix.txt '
        command += f'--transform SyN[0.1,3,0] '
        command += f'--metric CC[{fixed}, {moving},1,2] '
        command += f'--convergence [60x10,1e-6,10] '
        command += f'--shrink-factors 2x1 '
        command += f'--smoothing-sigmas 1x0vox '

        # Find correct motion mask
        try:
            # Check whether there are event related runs
            momaFile = f'{funcDir}/{sub}_{ses}_task-eventStim_run-001_nulled_moma.nii.gz'
            momaNii = nb.load(momaFile)
            command += f'-x {funcDir}/{sub}_{ses}_task-eventStim_run-001_nulled_moma.nii.gz'
        except:
            # Otherwise take moma of first run
            runMoma = sorted(glob.glob(f'{funcDir}/{sub}_ses-001_task-*_moma.nii.gz'))[0]
            command += f'-x {runMoma}'

        # Run command
        subprocess.run(command,shell=True)

        # Prepare command to apply transform and check quality
        command = 'antsApplyTransforms '
        command += f'--interpolation BSpline[5] '
        command += f'-d 3 -i {moving} '
        command += f'-r {fixed} '
        command += f'-t {regFolder}/registered1_1Warp.nii.gz '
        command += f'-t {regFolder}/registered1_0GenericAffine.mat '
        command += f'-o {moving.split(".")[0]}_registered-{ses}.nii'
        # Run command
        subprocess.run(command,shell=True)
