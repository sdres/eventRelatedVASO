'''

Read and save motion traces

'''

import ants
import os
import glob
import numpy as np
import pandas as pd


def my_ants_affine_to_distance(affine, unit):

    dx, dy, dz = affine[9:]

    rot_x = np.arcsin(affine[6])
    cos_rot_x = np.cos(rot_x)
    rot_y = np.arctan2(affine[7] / cos_rot_x, affine[8] / cos_rot_x)
    rot_z = np.arctan2(affine[3] / cos_rot_x, affine[0] / cos_rot_x)

    if unit == 'deg':
        deg = np.degrees
        R = np.array([deg(rot_x), deg(rot_y), deg(rot_z)])
    if unit == 'rad':
        R = np.array([rot_x, rot_y, rot_z])

    T = np.array([dx, dy, dz])

    return T, R


SUBS = ['sub-10']
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

# ============================================================================
# Read motion parameters from transformation files
# ============================================================================

for sub in SUBS:
    print(f'Working on {sub}')

    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*.nii.gz'))

    # Initialte list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1, 6):  # We had a maximum of 2 sessions
            if f'ses-00{i}' in run:
                sessions.append(f'ses-00{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    for ses in sessions:
        funcDir = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'

        # Look for individual runs (containing both nulled and notnulled images)
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-0*.nii.gz'))

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')

            # Set folder where motion traces were dumped
            motionDir = f'{funcDir}/motionParameters/{base}'

            for modality in ['nulled', 'notnulled']:

                # Get all transformation matrices
                mats = sorted(glob.glob(f'{motionDir}/{base}_{modality}_vol*'))

                Tr = []; Rt = []

                for i, mat in enumerate(mats):

                    localtxp = ants.read_transform(mat)
                    affine = localtxp.parameters

                    T, R = my_ants_affine_to_distance(affine, 'rad')

                    Tr.append(T)
                    Rt.append(R)

                # // Save motion traces intra-run as .csv
                Tr = np.asarray(Tr)
                Rt = np.asarray(Rt)

                data_dict = {'Tx': Tr[:, 0],
                             'Ty': Tr[:, 1],
                             'Tz': Tr[:, 2],
                             'Rx': Rt[:, 0],
                             'Ry': Rt[:, 1],
                             'Rz': Rt[:, 2]
                             }

                pd_ses = pd.DataFrame(data=data_dict)
                pd_ses.to_csv(os.path.join(motionDir, f'{base}_{modality}_motionTraces.csv'), index = False)
                pd_ses.to_csv(os.path.join(motionDir, f'{base}_{modality}_motionTraces.txt'), header=False, index=False)

# =================================================================
# Get motion summary
# =================================================================

for sub in SUBS:
    print(f'Working on {sub}')

    # =========================================================================
    # Look for sessions
    # Collect all runs across sessions (containing both nulled and notnulled images)
    allRuns = sorted(glob.glob(f'{ROOT}/{sub}/ses-*/func/{sub}_ses-0*_task-*run-0*.nii.gz'))

    # Initialte list for sessions
    sessions = []
    # Find all sessions
    for run in allRuns:
        for i in range(1,6):  # We had a maximum of 2 sessions
            if f'ses-00{i}' in run:
                sessions.append(f'ses-00{i}')

    # Get rid of duplicates
    sessions = sorted(set(sessions))
    print(f'Found data from sessions: {sessions}')

    for ses in sessions:
        funcDir = f'{ROOT}/derivativesTestTest/{sub}/{ses}/func'

        # Look for individual runs (containing both nulled and notnulled images)
        runs = sorted(glob.glob(f'{ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-*run-0*.nii.gz'))

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')


            # =========================================================================
            # Calculating framewise displacements

            fd = []
            timepoints = []
            subjects=[]
            mods = []

            for modality in ['nulled', 'notnulled']:

                # Set folder where motion traces were dumped
                runMotionDir = f'{funcDir}/motionParameters/{base}'

                # Load FD data
                data = pd.read_csv(os.path.join(runMotionDir, f'{base}_{modality}_motionTraces.csv'))

                TX = data['Tx'].to_numpy()
                TY = data['Ty'].to_numpy()
                TZ = data['Tz'].to_numpy()
                RX = data['Rx'].to_numpy()
                RY = data['Ry'].to_numpy()
                RZ = data['Rz'].to_numpy()

                for n in range(len(TX)-1):
                    fdVol = abs(TX[n]-TX[n+1])+abs(TY[n]-TY[n+1])+abs(TZ[n]-TZ[n+1])+abs((50*RX[n])-(50*RX[n+1]))+abs((50*RY[n])-(50*RY[n+1]))+abs((50*RZ[n])-(50*RZ[n+1]))
                    fd.append(fdVol)
                    timepoints.append(n)
                    subjects.append(sub)
                    mods.append(modality)

            FDs = pd.DataFrame({'subject': subjects, 'volume': timepoints, 'FD': fd, 'modality': mods})
            FDs.to_csv(os.path.join(runMotionDir, f'{base}_FDs.csv'), index=False)

            # =========================================================================
            # Formatting motion traces for plotting

            motionTraces = []
            motionNames = []
            volumes = []
            modalityList = []

            for modality in ['nulled', 'notnulled']:

                tmpBase = base + '_' + modality

                # Set folder where motion traces were dumped
                runMotionDir = f'{funcDir}/motionParameters/{base}'

                data = pd.read_csv(os.path.join(runMotionDir, f'{tmpBase}_motionTraces.csv'))

                for col in data.columns:
                    tmp = data[col].to_numpy()

                    for i, val in enumerate(tmp, start = 1):
                        motionTraces.append(val)
                        motionNames.append(f'{col} {modality}')
                        volumes.append(i)
                        modalityList.append(modality)

            data_dict = {
            'volume': volumes,
            'Motion': motionTraces,
            'name': motionNames,
            'modality': modalityList
            }

            pd_ses = pd.DataFrame(data = data_dict)
            pd_ses.to_csv(os.path.join(runMotionDir, f'{base}_motionSummary.csv'), index=False)