import numpy as np
import nibabel as nb
import subprocess
import glob
import os


root = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

for sub in ['sub-13']:

    maskFolder = f'{root}/derivatives/{sub}/ses-001/{sub}_masks'

    for nrLayers in [3, 11]:
        for focus in ['v1','s1']:
            if sub == 'sub-07':
                for stimType in ['blockStim', 'eventStim']:
                    os.system(f'/home/sebastian/git/laynii/LN2_LAYERS -rim {maskFolder}/{sub}_{focus}_rim_{stimType}.nii.gz -nr_layers {nrLayers} -output {maskFolder}/{sub}_{focus}_{nrLayers}layers_{stimType}')
            else:
                os.system(f'/home/sebastian/git/laynii/LN2_LAYERS -rim {maskFolder}/{sub}_{focus}_rim.nii.gz -nr_layers {nrLayers} -output {maskFolder}/{sub}_{focus}_{nrLayers}layers')




for sub in ['sub-12']:
    ses = 'ses-001'
    dataFolder = f'{root}/derivatives/{sub}/{ses}'

    runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

    # for run in runs:
    #     base = os.path.basename(run).rsplit('.', 2)[0][:-4]
    #     print(f'Processing run {base}')
    #
    #     # Get dims
    #     dataFile = f'{dataFolder}/{base}_T1w_N4Corrected.nii'
    #     dataNii = nb.load(dataFile)
    #     header = dataNii.header
    #     data = dataNii.get_fdata()
    #
    #     dims = header.get_zooms()
    #
    #     xdim = dims[0]
    #     ydim = dims[1]
    #     zdim = dims[2]
    #
    #     for modality in ['VASO', 'BOLD']:
    #         subprocess.run(f'/home/sebastian/abin/3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {dataFolder}/{base}_{modality}.feat/stats/zstat1_scaled.nii.gz -input {dataFolder}/{base}_{modality}.feat/stats/zstat1.nii.gz',shell=True)

    for stimType in ['blockStim', 'eventStimRandom']:
    # for stimType in ['blockStim', 'eventStim']:
    # for stimType in ['blockStim']:
    # for stimType in ['blockStim', 'blockStimLongTR', 'blockStimVisOnly']:
    # for stimType in ['blockStimVisOnly']:

        for modality in ['VASO', 'BOLD']:
            try:
                dataFile = f'{dataFolder}/{sub}_{ses}_task-{stimType}_secondLevel_{modality}.gfeat/cope1.feat/stats/zstat1.nii.gz'

                dataNii = nb.load(dataFile)
                header = dataNii.header
                data = dataNii.get_fdata()
            except:
                print('no secondLevel')
                continue



            dims = header.get_zooms()

            xdim = dims[0]
            ydim = dims[1]
            zdim = dims[2]

            try:
                for i in range(1,5):
                    subprocess.run(f'/home/sebastian/abin/3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {dataFolder}/{sub}_{ses}_task-{stimType}_secondLevel_{modality}.gfeat/cope{i}.feat/stats/zstat1_scaled.nii.gz -input {dataFolder}/{sub}_{ses}_task-{stimType}_secondLevel_{modality}.gfeat/cope{i}.feat/stats/zstat1.nii.gz',shell=True)
            except:
                subprocess.run(f'/home/sebastian/abin/3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {dataFolder}/{sub}_{ses}_task-{stimType}_secondLevel_{modality}.gfeat/cope1.feat/stats/zstat1_scaled.nii.gz -input {dataFolder}/{sub}_{ses}_task-{stimType}_secondLevel_{modality}.gfeat/cope1.feat/stats/zstat1.nii.gz',shell=True)



for sub in ['sub-11']:
    ses='ses-001'

    runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

    dataFolder = f'{root}/derivatives/{sub}/{ses}'
    outFolder = f'{root}/derivatives/{sub}/{ses}/upsampledFunctional'

    subprocess.run(f'mkdir {outFolder}', shell=True)

    for focus in ['v1', 's1']:

        if sub == 'sub-07':
            try:
                maskFile = f'{dataFolder}/{sub}_masks/{sub}_{focus}_{nrLayers}layers_blockStim_layers_equidist.nii'
            except:
                print(f'no mask found for {focus}. continuing...')
                coninue
        else:
            try:
                # find mask slice
                maskFile = f'{dataFolder}/{sub}_masks/{sub}_{focus}_3layers_layers_equidist.nii'
            except:
                print(f'no mask found for {focus}. continuing...')
                coninue

        maskNii = nb.load(maskFile)
        maskHeader = maskNii.header
        maskAffine = maskNii.affine
        mask = maskNii.get_fdata()

        for n in range(mask.shape[-1]):
            indexes = np.nonzero(mask[:,:,n])
            if len(indexes[0]) != 0:
                slice = n
        print(f'slice of mask is {slice}')

        for nrLayers in ['3','11']:
            if sub == 'sub-07':
                # for task in ['eventStim', 'blockStim']:
                for task in ['eventStim']:
                    maskFile = f'{dataFolder}/{sub}_masks/{sub}_{focus}_{nrLayers}layers_{task}_layers_equidist.nii'
                    maskNii = nb.load(maskFile)
                    maskHeader = maskNii.header
                    maskAffine = maskNii.affine
                    mask = maskNii.get_fdata()

                    for n in range(mask.shape[-1]):
                        indexes = np.nonzero(mask[:,:,n])
                        if len(indexes[0]) != 0:
                            slice = n
                    print(f'slice of mask is {slice}')
                    subprocess.run(f'fslroi {maskFile} {outFolder}/{sub}_{focus}_{nrLayers}layers_layers_{task}_equidist.nii 0 {mask.shape[0]} 0 {mask.shape[1]} {slice} 1',shell=True)

            else:
                maskFile = f'{dataFolder}/{sub}_masks/{sub}_{focus}_{nrLayers}layers_layers_equidist.nii'
                subprocess.run(f'fslroi {maskFile} {outFolder}/{sub}_{focus}_{nrLayers}layers_layers_equidist.nii 0 {mask.shape[0]} 0 {mask.shape[1]} {slice} 1',shell=True)

        for run in runs:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')

            for modality in ['VASO', 'BOLD']:
                dataFile = f'{dataFolder}/{base}_{modality}.nii.gz'
                dataNii = nb.load(dataFile)
                header = dataNii.header
                data = dataNii.get_fdata()
                subprocess.run(f'fslroi {dataFile} {outFolder}/{base}_{focus}_{modality}.nii.gz 0 {data.shape[0]} 0 {data.shape[1]} {slice} 1',shell=True)
                dims = header.get_zooms()

                xdim = dims[0]
                ydim = dims[1]
                zdim = dims[2]

                subprocess.run(f'/home/sebastian/abin/3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {outFolder}/{base}_{focus}_{modality}.nii.gz -input {outFolder}/{base}_{focus}_{modality}.nii.gz',shell=True)


## upsample QA

for sub in ['sub-05','sub-06','sub-07','sub-08','sub-09','sub-11','sub-12','sub-13','sub-14']:
    ses = 'ses-002'
    dataFolder = f'{root}/derivatives/{sub}/{ses}'

    runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')

        # Get dims
        dataFile = f'{dataFolder}/{base}_T1w_N4Corrected.nii'
        dataNii = nb.load(dataFile)
        header = dataNii.header
        data = dataNii.get_fdata()

        dims = header.get_zooms()

        xdim = dims[0]
        ydim = dims[1]
        zdim = dims[2]

        for modality in ['VASO', 'BOLD']:
            for measure in ['mean', 'tSNR', 'kurt', 'skew']:

                subprocess.run(f'/home/sebastian/abin/3dresample -dxyz {xdim/5} {ydim/5} {zdim} -rmode Cu -overwrite -prefix {dataFolder}/{base}_{modality}_{measure}_scaled.nii.gz -input {dataFolder}/{base}_{modality}_{measure}.nii.gz',shell=True)
