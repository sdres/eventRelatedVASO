import glob
import subprocess

path = '/Volumes/sDresBackUp/EVENTRELATED_PILOT/Nifti'

files = glob.glob(f'{path}/*/.DS_Store*')
import os


for root, dirs, files in os.walk(path):
    for name in files:
        if '.DS_Store' in name:
            print(name)