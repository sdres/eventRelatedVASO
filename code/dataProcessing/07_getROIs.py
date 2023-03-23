'''

Taking rim files as input and doing layerification.
Assumes that rim files were created based on upsampling x5 in x and y dir

'''

import glob
import subprocess
import os

ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
SUBS = ['sub-14']

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
        for focus in ['v1']:
            print(f'Working on {ses}')

            roiFolder =  f'{ROOT}/derivativesTestTest/{sub}/{ses}/func/rois'

            files = sorted(glob.glob(f'{roiFolder}/*rim-{focus}.nii*'))

            for file in files:
                base = file.split('.')[0]

                for nrLayers in [3,11]:
                    command = f'LN2_LAYERS '
                    command += f'-rim {file} '
                    command += f'-nr_layers {nrLayers} '
                    command += f'-equivol '
                    command += f'-output {base}_{nrLayers}layers'

                    subprocess.run(command, shell = True)
