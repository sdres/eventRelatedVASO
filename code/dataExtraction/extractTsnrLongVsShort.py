"""Extract tSNR values to compare long and short TR acquisitions"""

import nibabel as nb
import glob
import os
import pandas as pd

ROOT = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'
DERIVATIVES = f'{ROOT}/derivativesTestTest'

subs = ['sub-09', 'sub-11']

modalities = ['BOLD', 'VASO']

subList = []
modalityList = []
tSNRList = []
meanList = []
voxelList = []
runList = []
focusList = []
kurtList = []
skewList = []

for sub in subs:
    print(sub)

    # Find session in which long TRs were tested
    runs = sorted(glob.glob(f'{ROOT}/{sub}/ses-00*/func/{sub}_ses-00*_task-*LongTR*_cbv.nii.gz'))

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]

        if 'ses-001' in base:
            ses = 'ses-001'
        if 'ses-002' in base:
            ses = 'ses-002'

        for modality in modalities:
            for focus in ['v1', 's1']:
                try:
                    maskFile = f'{DERIVATIVES}/{sub}/{ses}/func/{sub}_masks/{sub}_{focus}_3layers_layers_equidist.nii'
                    maskNii = nb.load(maskFile)
                    maskData = maskNii.get_fdata()
                    idxMask = maskData > 0
                except:
                    print(f'no mask found for {focus}')
                    continue

                tSNRFile = f'{DERIVATIVES}/{sub}/{ses}/func/upsample/{base}_{modality}_tSNR_ups5x.nii'
                tSNRNii = nb.load(tSNRFile)
                tSNRData = tSNRNii.get_fdata()

                meanFile = f'{DERIVATIVES}/{sub}/{ses}/func/upsample/{base}_{modality}_mean_ups5x.nii'
                meanNii = nb.load(meanFile)
                meanData = meanNii.get_fdata()

                skewFile = f'{DERIVATIVES}/{sub}/{ses}/func/upsample/{base}_{modality}_skew_ups5x.nii'
                skewNii = nb.load(skewFile)
                skewData = skewNii.get_fdata()

                kurtFile = f'{DERIVATIVES}/{sub}/{ses}/func/upsample/{base}_{modality}_kurt_ups5x.nii'
                kurtNii = nb.load(kurtFile)
                kurtData = kurtNii.get_fdata()

                data1 = tSNRData[idxMask]
                data2 = meanData[idxMask]
                data3 = skewData[idxMask]
                data4 = kurtData[idxMask]

                for i, (value1, value2, value3, value4) in enumerate(zip(data1, data2, data3, data4)):
                    subList.append(sub)
                    modalityList.append(modality)
                    tSNRList.append(value1)
                    voxelList.append(i)
                    meanList.append(value2)
                    runList.append(base)
                    focusList.append(focus)
                    skewList.append(value3)
                    kurtList.append(value4)

# Store data in DataFrame
tSNRdata = pd.DataFrame({'subject': subList,
                         'modality': modalityList,
                         'tSNR': tSNRList,
                         'mean': meanList,
                         'voxel': voxelList,
                         'run': runList,
                         'focus': focusList,
                         'kurtosis': kurtList,
                         'skew': skewList
                         }
                        )

# Save to .csv file
tSNRdata.to_csv('results/qaLongVsShort.csv', sep=',', index=False)
