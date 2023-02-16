'''
Some things to Check
'''


import numpy as np
import nibabel as nb
import matplotlib.pyplot as plt

folder = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti/derivativesTestTest/sub-14/ses-001/func'



maskFile = f'{folder}/rois/sub-14_ses-001_T1w_testMask.nii.gz'
maskNii = nb.load(maskFile)
maskData = maskNii.get_fdata()
maskIdx = maskData == 1

runs = ['/Users/sebastiandresbach/data/eventRelatedVASO/Nifti/derivativesTestTest/sub-14/ses-001/func/sub-14_ses-001_task-blockStim_run-001_VASO.nii.gz',
        '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti/derivatives/sub-14/ses-001/sub-14_ses-001_task-blockStim_run-001_VASO.nii.gz']

# runs = ['/Users/sebastiandresbach/data/eventRelatedVASO/Nifti/derivativesTestTest/sub-14/ses-001/func/sub-14_ses-001_task-blockStim_run-001_notnulled_moco.nii']

states = ['new', 'old']

for state, run in enumerate(runs):
    nii = nb.load(run)
    data = nii.get_fdata()

    ts = []

    for tp in range(data.shape[-1]):

        tmp = np.mean(data[...,tp][maskData.astype('bool')])
        ts.append(tmp)

    plt.plot(ts,label=f'{states[state]}')
    plt.legend()
