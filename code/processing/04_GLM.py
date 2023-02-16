'''

Running first level GLM in FSL using Nilearn

'''

import nibabel as nb
import nilearn
import numpy as np
from nilearn.glm.first_level import make_first_level_design_matrix
import glob
import pandas as pd
from nilearn.glm.first_level import FirstLevelModel
import os
import re
import sys

sys.path.append('./code')

from find_tr import *


drift_model = 'Cosine'  # We use a discrete cosine transform to model signal drifts.
high_pass = .01  # The cutoff for the drift model is 0.01 Hz.
hrf_model = 'spm'  # The hemodynamic response function is the SPM canonical one.

ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

SUBS = ['sub-14']
# SUBS = ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']


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

        # Set folder for session outputs
        sesFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'
        statFolder = f'{sesFolder}/statMaps'

        # Create folder if it does not exist
        if not os.path.exists(statFolder):
            os.makedirs(statFolder)
            print("statMap directory is created")

        runTypes = []
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*_run-00*_cbv.nii.gz'))
        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-12].split('-')[-1]
            if base not in runTypes:
                runTypes.append(base)

        for runType in runTypes:
            print(f'Processing {runType}')
            # if runType == 'blockStimRandom':
            # if 'eventS' in runType:
            #     print('skipping')
            #     continue

            # Look for individual runs within session (containing both nulled and notnulled images)
            runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-00*_cbv.nii.gz'))

            for modality in ['VASO', 'BOLD']:  # Loop over BOLD and VASO
                print(f'Processing {modality}')

                niiFiles = []
                design_matrices = []

                for run in runs:

                    base = os.path.basename(run).rsplit('.', 2)[0][:-4]

                    tr = findTR(f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log')

                    niiFile = f'{sesFolder}/{base}_{modality}.nii.gz'
                    niiFiles.append(niiFile)

                    nii = nb.load(niiFile)
                    data = nii.get_fdata()
                    nVols = data.shape[-1]
                    frame_times = np.arange(nVols) * tr

                    events = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/{base}_events.tsv', sep = ' ')

                    design_matrix = make_first_level_design_matrix(
                        frame_times,
                        events,
                        hrf_model=hrf_model,
                        drift_model = None,
                        high_pass= high_pass
                        )
                    design_matrices.append(design_matrix)

                fmri_glm = FirstLevelModel(mask_img = False, drift_model = None)

                fmri_glm = fmri_glm.fit(niiFiles, design_matrices = design_matrices)

                contrast_matrix = np.eye(design_matrix.shape[1])
                basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])

                if not ('Random' in runType) and not ('VisOnly' in runType):
                    print('only contrast is visiotactile')
                    if modality == 'BOLD':
                        contrasts = {'visiotactile': + basic_contrasts['visiotactile']
                            }
                    if modality == 'VASO':
                        contrasts = {'visiotactile': - basic_contrasts['visiotactile']
                            }

                if 'Random' in runType:
                    print('both visual and visiotactile')
                    if modality == 'BOLD':
                        contrasts = {'visiotactile': + basic_contrasts['visiotactile'],
                                     'visual': + basic_contrasts['visual']                            }

                    if modality == 'VASO':
                        contrasts = {'visiotactile': - basic_contrasts['visiotactile'],
                                     'visual': - basic_contrasts['visual']                            }

                if 'VisOnly' in runType:
                    print('only contrast is visual')

                    if modality == 'BOLD':
                        contrasts = {'visual': + basic_contrasts['visual']
                            }
                    if modality == 'VASO':
                        contrasts = {'visual': - basic_contrasts['visual']
                            }

                # Iterate on contrasts
                for contrast_id, contrast_val in contrasts.items():
                    print(f'Computing contrast: {contrast_id}')
                    print(f'With matrix: {contrast_val}')
                    # compute the contrasts
                    z_map = fmri_glm.compute_contrast(
                        contrast_val, output_type='z_score')
                    nb.save(z_map, f'{statFolder}/{sub}_{runType}_{modality}_{contrast_id}.nii')


# =============================================================================
# Estimate FIR
# =============================================================================

print('\nRunning FIR analysis')

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

        # Set folder for session outputs
        sesFolder = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'
        statFolder = f'{sesFolder}/statMaps'

        # Create folder if it does not exist
        if not os.path.exists(statFolder):
            os.makedirs(statFolder)
            print("statMap directory is created")

        runTypes = []
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-event*_run-00*_cbv.nii.gz'))
        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-12].split('-')[-1]
            if base not in runTypes:
                runTypes.append(base)

        for runType in runTypes:
            print(f'Processing {runType}')

            # Look for individual runs within session (containing both nulled and notnulled images)
            runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-{runType}_run-00*_cbv.nii.gz'))

            for modality in ['VASO', 'BOLD']:
                print(f'Processing {modality}')

                niiFiles = []
                design_matrices = []

                for run in runs:

                    base = os.path.basename(run).rsplit('.', 2)[0][:-4]

                    tr = findTR(f'{ROOT}/derivatives/{sub}/{ses}/events/{base}.log')

                    niiFile = f'{sesFolder}/{base}_{modality}.nii.gz'
                    niiFiles.append(niiFile)

                    nii = nb.load(niiFile)
                    data = nii.get_fdata()
                    nVols = data.shape[-1]
                    frame_times = np.arange(nVols) * tr

                    events = pd.read_csv(f'{ROOT}/{sub}/{ses}/func/{base}_events.tsv', sep = ' ')

                    design_matrices.append(events)

                fmri_glm = FirstLevelModel(tr, hrf_model='fir', fir_delays=np.arange(0,11),mask_img = False, drift_model=None)

                fmri_glm = fmri_glm.fit(niiFiles, events=design_matrices)

                design_matrix = fmri_glm.design_matrices_[0]

                contrast_matrix = np.eye(design_matrix.shape[1])

                contrasts = dict([(column, contrast_matrix[i])
                                  for i, column in enumerate(design_matrix.columns)])

                basic_contrasts = dict([(column, contrast_matrix[i])
                            for i, column in enumerate(design_matrix.columns)])

                contrasts = {}


                contrastTypes = []

                if not ('Random' in runType) and not ('VisOnly' in runType):
                    # print('only contrast is visiotactile')
                    contrastTypes.append('visiotactile')

                if 'Random' in runType:
                    # print('both visual and visiotactile')
                    contrastTypes.append('visiotactile')
                    contrastTypes.append('visual')

                if 'VisOnly' in runType:
                    print('only contrast is visual')
                    contrastTypes.append('visual')



                for contrastType in contrastTypes:
                    for delay in np.arange(0,11):
                        if modality == 'VASO':
                            contrasts[f'{contrastType}{delay:02d}'] = - basic_contrasts[f'{contrastType}_delay_{delay}']
                        if modality == 'BOLD':
                            contrasts[f'{contrastType}{delay:02d}'] = + basic_contrasts[f'{contrastType}_delay_{delay}']



                for contrast_id, contrast_val in contrasts.items():
                    # compute the contrasts
                    z_map = fmri_glm.compute_contrast(
                        contrast_val, output_type='effect_size')
                    nb.save(z_map, f'{statFolder}/{sub}_FIR_{modality}_{contrast_id}.nii.gz')
