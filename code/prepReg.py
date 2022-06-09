import subprocess
import glob

featFolders = sorted(glob.glob('/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/sub-09/ses-001/*feat'))

for folder in featFolders:
    command = f'mkdir {folder}/reg'
    subprocess.run(command, shell=True)
    command = f'cp $FSLDIR/etc/flirtsch/ident.mat {folder}/reg/example_func2standard.mat'
    subprocess.run(command, shell=True)
    command = f'cp {folder}/mean_func.nii.gz {folder}/reg/standard.nii.gz'
    subprocess.run(command, shell=True)
