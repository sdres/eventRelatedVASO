import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set folders
gitProjectDir = Path.cwd()
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

# ============================================================================
# Set global plotting parameters

plt.style.use('dark_background')
PALETTE = {'notnulled': 'tab:orange', 'nulled': 'tab:blue'}

LW = 2
motionPalette = ['Set1', 'Set2']

# =================================================================
# Make directory
outFolder = f'{gitProjectDir}/results/motionSummary'
# Create folder if it does not exist
if not os.path.exists(outFolder):
    os.makedirs(outFolder)
    print("Motion directory is created")

# =================================================================
# Plotting
# =================================================================

SUBS = ['sub-12']

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
            runMotionDir = f'{funcDir}/motionParameters/{base}'

            # Load motion data
            data = pd.read_csv(os.path.join(runMotionDir, f'{base}_motionSummary.csv'))

            # Initialize plot
            fig, axes = plt.subplots(1, 2, sharex=True, figsize=(30, 6))
            plt.suptitle(f'{base} Motion Summary', fontsize=24)

            # Plotting translation and rotation on different axes
            for i, motionType in enumerate(['T', 'R']):

                # Only plot legend for rotation plot (will be on the right)
                # Colors are the same for both plots

                if motionType == 'T':
                    legend = False
                if motionType == 'R':
                    legend = True

                for modality, cmap in zip(['nulled', 'notnulled'], ['Set1', 'Set2']):

                    if modality == 'nulled':
                        tmpData = data.loc[(data['name'].str.contains(motionType))
                                           & (-data['name'].str.contains('notnulled'))]
                    else:
                        tmpData = data.loc[(data['name'].str.contains(motionType))
                                           & (data['name'].str.contains('notnulled'))]

                    sns.lineplot(ax=axes[i],
                                 x='volume',
                                 y='Motion',
                                 data=tmpData,
                                 hue='name',
                                 palette=cmap,
                                 linewidth=LW,
                                 legend=legend
                                 )

            # Set y label
            axes[0].set_ylabel("Translation [mm]", fontsize=24)
            axes[1].set_ylabel("Rotation [radians]", fontsize=24)

            # Colors are the same for both plots, so we only need one legend
            axes[1].legend(fontsize=20,
                           loc='center left',
                           bbox_to_anchor=(1, 0.5)
                           )

            # X label is the same for both plots
            for j in range(2):
                axes[j].tick_params(axis='both', labelsize=20)
                axes[j].set_xlabel("Volume", fontsize=24)

            # Save figure
            plt.savefig(f'{outFolder}/{base}_motion.jpg', bbox_inches='tight', pad_inches=0)
            plt.show()

            # =========================================================================
            # Plotting FDs

            FDs = pd.read_csv(os.path.join(runMotionDir, f'{base}_FDs.csv'))

            plt.figure(figsize=(20, 5))

            sns.lineplot(data=FDs,
                         x='volume',
                         y='FD',
                         hue='modality',
                         linewidth=LW,
                         palette=PALETTE
                         )

            if np.max(FDs['FD']) < 0.9:
                plt.ylim(0, 1)

            plt.axhline(0.9, color='gray', linestyle='--')
            plt.ylabel('FD [mm]', fontsize=24)
            plt.xlabel('Volume', fontsize=24)

            plt.legend(fontsize=20,
                       loc='center left',
                       bbox_to_anchor=(1, 0.5)
                       )

            plt.xticks(fontsize=20)
            plt.yticks(fontsize=20)

            plt.title(base, fontsize=24, pad=20)
            plt.savefig(f'{outFolder}/{base}_FDs.png', bbox_inches='tight', pad_inches=0)
            plt.show()
