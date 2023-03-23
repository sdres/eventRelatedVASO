"""Downsample ROIs to native resolution"""

import nibabel as nb
import subprocess
import glob
import os

ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
subs = ['sub-05', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-11', 'sub-12', 'sub-13', 'sub-14']

for sub in subs:
    print(f'Working on {sub}')

    # Set folder for subject
    subFolder = f'{ROOT}/derivativesTestTest/{sub}'

    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
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

        sesFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}'
        outFolder = f'{sesFolder}/func/upsample'

        # ======================================================================
        # Get dims
        if not sub == 'sub-07':
            dataFile = f'{sesFolder}/func/{sub}_{ses}_T1w.nii'
            base = os.path.basename(dataFile).rsplit('.', 2)[0]

        else:
            dataFile = f'{sesFolder}/func/{sub}_{ses}_eventStim_T1w.nii'
            base = os.path.basename(dataFile).rsplit('.', 2)[0]

        dataNii = nb.load(dataFile)
        header = dataNii.header
        data = dataNii.get_fdata()

        dims = header.get_zooms()

        xdim = dims[0]
        ydim = dims[1]
        zdim = dims[2]

        # Get masks
        if not sub == 'sub-07':
            masks = glob.glob(f'{sesFolder}/func/{sub}_masks/{sub}_*_rim.nii.gz')
        if sub == 'sub-07':
            masks = glob.glob(f'{sesFolder}/func/{sub}_masks/{sub}_*_rim*Stim.nii.gz')

        for mask in masks:
            base = os.path.basename(mask).split('.')[0]
            outName = f'{sesFolder}/func/{sub}_masks/{base}_down.nii.gz'
            # Downsample
            command = f'3dresample -dxyz {xdim} {ydim} {zdim} -rmode Cu -overwrite -prefix {outName} -input {mask}'
            subprocess.run(command, shell=True)
            # Threshold and binarize
            command2 = f'fslmaths {outName} -thr 3 -bin {outName}'
            subprocess.run(command2, shell=True)
