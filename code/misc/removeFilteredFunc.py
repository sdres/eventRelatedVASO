"""Remove filtered func data to save space"""

import subprocess
import glob
# Set some folder names
ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
DERIVATIVES = f'{ROOT}/derivativesTestTest'

# Find filtered func data

files = sorted(glob.glob(f'{DERIVATIVES}/sub-*/ses-*/func/*.feat/filtered_func_data.nii.gz'))

for file in files:
    command = f'rm {file}'
    subprocess.run(command, shell=True)
