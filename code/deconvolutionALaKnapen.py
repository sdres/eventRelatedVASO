import nideconv
import nibabel as nb
import numpy as np
import glob
import pandas as pd
import os

import seaborn as sns
import matplotlib.pyplot as plt

root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

data = nb.load('/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/sub-05/ses-001/sub-05_ses-001_task-eventStim_run-001_v1_BOLD_layers_ts.nii').get_fdata()

data = data[-1,:,:]

data.shape

test = np.reshape(data, -1)
test = test.T

test.shape


onsetsLong = pd.read_csv(f'{root}/derivatives/sub-05/ses-001/events/sub-05_ses-001_task-eventStim_run-001_longITI.txt',delimiter=' ', names=['onset','duration','mod'])

onsetsShort = pd.read_csv(f'{root}/derivatives/sub-05/ses-001/events/sub-05_ses-001_task-eventStim_run-001_longITI.txt',delimiter=' ', names=['onset','duration','mod'])

rf =nideconv.ResponseFitter(input_signal=test,sample_rate=1/1.3)

rf.add_event(event_name='longITI',
             onsets=onsetsLong['onset'],
             interval=[0, 19.5])

rf.add_event(event_name='shortITI',
             onsets=onsetsShort.onset,
             interval=[0, 19.5])


sns.heatmap(rf.X)

print(rf.X)


rf.fit()
print(rf.betas)

tc =rf.get_timecourses()
print(tc)

# sns.set_palette(palette)
rf.plot_timecourses()
plt.suptitle('Linear deconvolution using GLM and FIR')
plt.title('')
plt.legend()
