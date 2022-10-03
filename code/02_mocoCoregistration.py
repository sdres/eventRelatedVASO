'''

Do motion correction and register runs

'''

import ants
import os
import glob
from nipype.interfaces import afni
import nibabel as nb
import numpy as np
import subprocess
from IPython.display import clear_output
import nipype.interfaces.fsl as fsl
import itertools
import pandas as pd
import sys

sys.path.append('./code')

from computeT1w import *

# Set base directory
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

# Set subjects to work on
SUBS = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']
SUBS = ['sub-05']
# SUBS = ['sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']

for sub in SUBS:
    print(f'Working on {sub}')

    # Set folder for subject
    subFolder = f'{ROOT}/derivativesTest/{sub}'
    # Create folder if it does not exist
    if not os.path.exists(subFolder):
        os.makedirs(subFolder)
        print("Session directory is created")

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

        # Look for individual runs within session (containing both nulled and notnulled images)
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

        # Set folder for session outputs
        outFolder = f'{ROOT}/derivativesTest/{sub}/{ses}'
        # Create folder if it does not exist
        if not os.path.exists(outFolder):
            os.makedirs(outFolder)
            print("Session directory is created")

        # Set folder for motion traces
        motionDir = f'{outFolder}/motionParameters'
        # Make folder to dump motion traces if it does not exist
        if not os.path.exists(motionDir):
            os.makedirs(motionDir)
            print("Motion directory is created")

        for run in runs:
            # Get base name of run
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')

            # Set folder for motion traces
            runMotionDir = f'{motionDir}/{base}'
            # Make folder to dump motion traces if it does not exist
            if not os.path.exists(runMotionDir):
                os.makedirs(runMotionDir)
                print("Runwise motion directory is created")


            for start, modality in enumerate(['notnulled', 'nulled']):
                print(f'Starting with {modality}')

                # Load timeseries containing nulled and notnulled
                nii = nb.load(run)
                # Get header and affine
                header = nii.header
                affine = nii.affine
                # Load data as array
                dataComplete = nii.get_fdata()

                # Separate nulled and notnulled data
                data = dataComplete[:,:,:, start:-2:2]  # Start is defined by "enumerate" above. 0 for notnulled, 1 for nulled. Here, I also get rid of the noise maps

                # Make new nii and save
                img = nb.Nifti1Image(data, header=header, affine=affine)
                nb.save(img, f'{outFolder}/{base}_{modality}.nii')

                # Make reference image
                reference = np.mean(data[:,:,:,4:6], axis = -1)

                # And save it
                img = nb.Nifti1Image(reference, header=header, affine=affine)
                nb.save(img, f'{outFolder}/{base}_{modality}_reference.nii')

                # Load reference in antsPy style
                fixed = ants.image_read(f'{outFolder}/{base}_{modality}_reference.nii')

                # Create motion mask
                mask = ants.get_mask(fixed, cleanup=2)
                ants.image_write(mask, f'{outFolder}/{base}_{modality}_moma.nii')

                # Load data in antsPy style
                ts = ants.image_read(f'{outFolder}/{base}_{modality}.nii')

                # Perform motion correction
                corrected = ants.motion_correction(ts, fixed = fixed, mask = mask)
                ants.image_write(corrected['motion_corrected'], f'{outFolder}/{base}_{modality}_moco.nii')

                # Save transformation matrix for later
                for vol, matrix in enumerate(corrected['motion_parameters']):
                    mat = matrix[0]
                    os.system(f"cp {mat} {runMotionDir}/{base}_{modality}_vol{vol:03d}.mat")

            # =========================================================================
            # Compute T1w image in EPI space within run

            t1w = computeT1w(f'{outFolder}/{base}_nulled_moco.nii', f'{outFolder}/{base}_notnulled_moco.nii')
            header = nb.load(f'{outFolder}/{base}_nulled_moco.nii').header
            affine = nb.load(f'{outFolder}/{base}_nulled_moco.nii').affine
            # And save the image
            img = nb.Nifti1Image(t1w, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{base}_T1w.nii')




############################################################################
############# Here, the coregistration of multiple runs starts #############
############################################################################

for sub in SUBS:
    print(f'Working on {sub}')

    # Set folder for subject
    subFolder = f'{ROOT}/derivativesTest/{sub}'

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

        # Set folder for session outputs
        outFolder = f'{ROOT}/derivativesTest/{sub}/{ses}'

        # Set folder for motion traces
        motionDir = f'{outFolder}/motionParameters'

        # Collectall runs within session (containing both nulled and notnulled images)
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_*_task-*run-00*_cbv.nii.gz'))

        try: # trying to register to first eventStim run

            referenceRun = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_ses-00*_task-event*run-00*_cbv.nii.gz'))[0]
            refBase = os.path.basename(referenceRun).rsplit('.', 2)[0][:-4]
            runs.remove(referenceRun)
            print(f'Registering all runs to {refBase}')

        except: # if not possible, register to first run
            referenceRun = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_ses-00*_task-*run-00*_cbv.nii.gz'))[0]
            refBase = os.path.basename(referenceRun).rsplit('.', 2)[0][:-4]
            runs.remove(referenceRun)
            print(f'Registering all runs to {refBase}')


        fixed = ants.image_read(f'{outFolder}/{refBase}_T1w.nii')
        mask = ants.image_read(f'{outFolder}/{refBase}_nulled_moma.nii')

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')

            # Define moving image
            moving = ants.image_read(f'{outFolder}/{base}_T1w.nii')
            # Do registration
            mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid', mask=mask)
            # Copy transform file for later
            os.system(f"cp {mytx['fwdtransforms'][0]} {outFolder}/{base}_T1w_registered-{refBase}.mat")
            # Apply registration
            mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'], interpolator='bSpline')
            # Save transformed image
            ants.image_write(mywarpedimage, f'{outFolder}/{base}_T1w_registered-{refBase}.nii')


            # Load transform we just saved
            transform1 = f'{outFolder}/{base}_T1w_registered-{refBase}.mat'

            # Loop over nulled and notnulled
            for start, modality in enumerate(['notnulled', 'nulled']):
                print(f'Starting with {modality}')

                # Load timeseries containing nulled and notnulled
                nii = nb.load(run)
                # Get header and affine
                header = nii.header
                affine = nii.affine
                # Load data as array
                dataComplete = nii.get_fdata()

                # Separate nulled and notnulled data
                data = dataComplete[:,:,:, start:-2:2]  # Start is defined by "enumerate" above. 0 for notnulled, 1 for nulled. Here, I also get rid of the noise maps

                # Separate volumes
                for i in range(data.shape[-1]):
                    if i <= 2: # Overwrite first 3 volumes with volumes 4,5 and 6
                        vol = data[:,:,:,i+3]
                    else:
                        vol = data[:,:,:,i]

                    img = nb.Nifti1Image(vol, header=header, affine=affine)
                    nb.save(img, f'{outFolder}/{base}_{modality}_vol{i:03d}.nii')


                for i in range(data.shape[-1]):
                    # Load volume
                    moving = ants.image_read(f'{outFolder}/{base}_{modality}_vol{i:03d}.nii')
                    # Load within run transofrmation of that volume
                    transform2 = f'{motionDir}/{base}/{base}_{modality}_vol{i:03d}.mat'

                    # Apply both transformations
                    mywarpedimage = ants.apply_transforms(fixed = fixed,
                                                          moving = moving,
                                                          transformlist = [transform1, transform2],
                                                          interpolator = 'bSpline'
                                                          )
                    # Save warped image
                    ants.image_write(mywarpedimage, f'{outFolder}/{base}_{modality}_vol{i:03d}.nii')

                # Make array of data shape to assample warped images
                newData = np.zeros(data.shape)

                # Loop over volumes to fill new data
                for i in range(data.shape[-1]):
                    vol = nb.load(f'{outFolder}/{base}_{modality}_vol{i:03d}.nii').get_fdata()
                    newData[:,:,:,i] = vol

                # Save new data
                img = nb.Nifti1Image(newData, header=header, affine=affine)
                nb.save(img, f'{outFolder}/{base}_{modality}_moco_reg.nii')
                # Delete individual volumes
                os.system(f'rm {outFolder}/{base}_{modality}_vol*.nii')

            # =========================================================================
            # Compute T1w image in EPI space within registered run

            t1w = computeT1w(f'{outFolder}/{base}_nulled_moco_reg.nii', f'{outFolder}/{base}_notnulled_moco_reg.nii')
            header = nb.load(f'{outFolder}/{base}_nulled_moco_reg.nii').header
            affine = nb.load(f'{outFolder}/{base}_nulled_moco_reg.nii').affine
            # And save the image
            img = nb.Nifti1Image(t1w, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{base}_reg_T1w.nii')


# Make overall t1w
dataFiles = sorted(glob.glob(f'{outFolder}/*_moco_reg.nii'))
for modality in ['nulled', 'notnulled']:
    dataFiles.append(f'{outFolder}/{refBase}_{modality}_moco.nii')

for i, file in enumerate(dataFiles):
    # Load nulled motion corrected timeseries
    nii = nb.load(file)
    data = nii.get_fdata()

    if i == 0:
        combined = data
    else:
        # Concatenate nulled and notnulled timeseries
        combined = np.concatenate((combined,data), axis=3)


    stdDev = np.std(combined, axis = 3)

    #Compute mean
    mean = np.mean(combined, axis = 3)
    # Compute variation
    cvar = stdDev/mean
    # Take inverse
    cvarInv = 1/cvar

# And save the image
img = nb.Nifti1Image(cvarInv, header=header, affine=affine)
nb.save(img, f'{outFolder}/{sub}_{ses}_T1w.nii')
