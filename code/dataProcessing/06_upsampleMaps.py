import numpy as np
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
        for i in range(1,3):  # We had a maximum of 2 sessions
            if f'ses-00{i}' in run:
                sessions.append(f'ses-00{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    # Loop over sessions
    for ses in sessions:
        print(f'Working on {ses}')

        sesFolder =  f'{ROOT}/derivativesTestTest/{sub}/{ses}'
        outFolder = f'{sesFolder}/func/upsample'
        # Create folder if it does not exist
        if not os.path.exists(outFolder):
            os.makedirs(outFolder)
            print("Output directory is created")

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

        # # ======================================================================
        # # Upsample t1w
        # subprocess.run(f'3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {outFolder}/{base}_ups5x.nii -input {dataFile}' ,shell=True)
        #
        # # # Upsample registered Anat
        # # dataFile = glob.glob(f'{sesFolder}/anat/{sub}_brain_registered*.nii')[0]
        # # base = os.path.basename(dataFile).rsplit('.', 2)[0]
        # # subprocess.run(f'3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {outFolder}/{base}_ups5x.nii -input {dataFile}' ,shell=True)
        #
        #
        # # Set folder for session outputs
        # statFolder = f'{sesFolder}/func/statMaps'
        #
        # # Get statistical maps
        # maps = sorted(glob.glob(f'{statFolder}/*'))
        #
        # for map in maps:
        #     base = os.path.basename(map).rsplit('.', 2)[0]
        #
        #     subprocess.run(f'3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {outFolder}/{base}_ups5x.nii -input {map}' ,shell=True)

        ## upsample QA
        for measure in ['mean', 'tSNR', 'kurt', 'skew']:
            maps = sorted(glob.glob(f'{sesFolder}/func/*{measure}*'))
            for map in maps:
                base = os.path.basename(map).rsplit('.', 2)[0]

                subprocess.run(f'3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {outFolder}/{base}_ups5x.nii.gz -input {map}' ,shell=True)
