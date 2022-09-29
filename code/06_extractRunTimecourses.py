import numpy as np
import glob
import nibabel as nb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import interp1d
import os

subs = ['sub-05']
ses = 'ses-001'
root = '/Users/sebastiandresbach/data/eventRelatedVASO/Nifti'

modalities = ['BOLD', 'VASO']

for focus in ['v1']:

    for sub in subs:
        outFolder = f'{root}/derivatives/{sub}/{ses}/upsampledFunctional'
        if sub == 'sub-07':
            mask = nb.load(f'{outFolder}/{sub}_{focus}_3layers_layers_eventStim_equidist.nii.gz').get_fdata()
        else:
            mask = nb.load(f'{outFolder}/{sub}_{focus}_3layers_layers_equidist.nii.gz').get_fdata()
        idx_layers = np.unique(mask.astype("int"))
        idx_layers = idx_layers[1:]
        idxMask = mask != 0


        print(sub)
        outFolder = f'{root}/derivatives/{sub}/{ses}/upsampledFunctional'


        runsAll = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-*_run-00*_cbv.nii.gz'))

        sessions = []
        for run in runsAll:

            if 'ses-001' in run:
                if not any('ses-001' in s for s in sessions):
                    sessions.append('ses-001')
            if 'ses-002' in run:
                if not any('ses-002' in s for s in sessions):
                    sessions.append('ses-002')

        for ses in sessions:
            if sub == 'sub-09' and focus == 's1Focus' and ses=='ses-002':
                continue

            runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-event*_run-00*_cbv.nii.gz'))

            for run in runs[1:]:
                if 'Long' in run:
                    continue

                base = os.path.basename(run).rsplit('.', 2)[0][:-4]


                for modality in modalities:

                    print(f'processing {base} {modality}')

                    run = f'{outFolder}/{base}_{focus}_{modality}.nii.gz'

                    Nii = nb.load(run)
                    # As before, get the data as an array.
                    data = Nii.get_fdata()
                    # load the nifty-header to get some meta-data.
                    header = Nii.header
                    affine = Nii.affine

                    # Or the TR, which is the 4th position of get_zooms().
                    tr = header.get_zooms()[3]

                     # Get scan duration in s
                    runTime = data.shape[-1]*tr

                    layerTS = np.zeros((3,1,1,data.shape[-1]))
                    layerTS.shape


                    for j in idx_layers:  # Compute bin averages

                        layerRoi = mask == j

                        print(data.shape)
                        print(layerRoi.shape)


                        mask_mean = np.mean(data[layerRoi],axis=0)[0]
                        df = pd.DataFrame({'signal':mask_mean})
                        df.to_csv(f'{root}/derivatives/{sub}/{ses}/{base}_{focus}_{modality}_layer{j:02d}.csv', index=False)

                        for i, tp in enumerate(mask_mean):
                            layerTS[j-1,0,0,i] = tp

                    nii = nb.Nifti1Image(layerTS, header = header, affine = affine)
                    nb.save(nii, f'{root}/derivatives/{sub}/{ses}/{base}_{focus}_{modality}_layers_ts.nii')


                    mask_mean = np.mean(data[idxMask],axis=0)[0]
                    df = pd.DataFrame({'signal':mask_mean})
                    df.to_csv(f'{root}/derivatives/{sub}/{ses}/{base}_{focus}_{modality}.csv', index=False)

                    mask_mean = mask_mean.reshape((1,1,1,mask_mean.shape[-1]))
                    nii = nb.Nifti1Image(mask_mean, header = header, affine = affine)
                    nb.save(nii, f'{root}/derivatives/{sub}/{ses}/{base}_{focus}_{modality}_ts.nii')





# For Vessel ROI
for focus in ['']:

    for sub in subs:
        outFolder = f'{root}/derivatives/{sub}/{ses}/upsampledFunctional'

        mask = nb.load(f'{outFolder}/{sub}_3layers_layers_equidist.nii.gz').get_fdata()
        idx_layers = np.unique(mask.astype("int"))
        idx_layers = idx_layers[1:]
        idxMask = mask != 0


        print(sub)
        outFolder = f'{root}/derivatives/{sub}/{ses}/upsampledFunctional'


        runsAll = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-eventStim*_run-00*_cbv.nii.gz'))

        sessions = []
        for run in runsAll:

            if 'ses-001' in run:
                if not any('ses-001' in s for s in sessions):
                    sessions.append('ses-001')
            if 'ses-002' in run:
                if not any('ses-002' in s for s in sessions):
                    sessions.append('ses-002')

        for ses in sessions:
            if sub == 'sub-09' and focus == 's1Focus' and ses=='ses-002':
                continue

            runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-eventSti*_run-00*_cbv.nii.gz'))

            for run in runs:
                if 'Long' in run:
                    continue

                base = os.path.basename(run).rsplit('.', 2)[0][:-4]


                for modality in modalities:

                    print(f'processing {base}')

                    run = f'{outFolder}/{base}_{modality}.nii.gz'

                    Nii = nb.load(run)
                    # As before, get the data as an array.
                    data = Nii.get_fdata()
                    # load the nifty-header to get some meta-data.
                    header = Nii.header
                    affine = Nii.affine

                    # Or the TR, which is the 4th position of get_zooms().
                    tr = header.get_zooms()[3]

                     # Get scan duration in s
                    runTime = data.shape[-1]*tr

                    vesselTS = np.zeros((1,1,1,data.shape[-1]))


                    for j in idx_layers:  # Compute bin averages

                        layerRoi = mask == j


                        mask_mean = np.mean(data[layerRoi],axis=0)[0]
                        # df = pd.DataFrame({'signal':mask_mean})
                        # df.to_csv(f'{root}/derivatives/{sub}/ses-001/{base}_{modality}_layer{j:02d}.csv', index=False)

                        for i, tp in enumerate(mask_mean):
                            vesselTS[0,0,0,i] = tp

                    nii = nb.Nifti1Image(vesselTS, header = header, affine = affine)
                    nb.save(nii, f'{root}/derivatives/{sub}/{ses}/{base}_{modality}_vessel_ts.nii')

                    #
                    # mask_mean = np.mean(data[idxMask],axis=0)[0]
                    # df = pd.DataFrame({'signal':mask_mean})
                    # df.to_csv(f'{root}/derivatives/{sub}/ses-001/{base}_{modality}.csv', index=False)
                    #
                    # mask_mean = mask_mean.reshape((1,1,1,mask_mean.shape[-1]))
                    # nii = nb.Nifti1Image(mask_mean, header = header, affine = affine)
                    # nb.save(nii, f'{root}/derivatives/{sub}/ses-001/{base}_{modality}_ts.nii')
