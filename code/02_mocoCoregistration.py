import ants
import os
import glob
from nipype.interfaces import afni
import nibabel as nb
import numpy as np
import subprocess
from IPython.display import clear_output
import nipype.interfaces.fsl as fsl
import itertools
import pandas as pd

root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'

for sub in ['sub-09']:
    ses = 'ses-001'

    # look for individual runs (containing both nulled and notnulled images)
    runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))
    # make folder to dump motion traces
    outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}'
    os.system(f'mkdir {outFolder}/motionParameters')

    for run in runs[-2:-1]:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')
        # make folder to dump motion traces for the run
        os.system(f'mkdir {outFolder}/motionParameters/{base}')

        # initiate lists for motion traces
        lstsubMot = [] # List for the motion value
        lstsubMot_Nme = [] # List for the motion name (e.g. translation in x direction)
        lstTR_sub = [] # List for name of subject. technically not needed because we are only doing it run by run
        modalityList = [] # List for nulled/notnulled

        for start, modality in enumerate(['notnulled', 'nulled']):
            print(modality)
            # Load timeseries containing nulled and notnulled
            nii = nb.load(run)
            # get header and affine
            header = nii.header
            affine = nii.affine
            # Load data as array
            dataComplete = nii.get_fdata()

            # separate nulled and notnulled data
            data = dataComplete[:,:,:,start:-2:2] # Start is defined by "enumerate" above. 0 for notnulled, 1 for nulled. Here, I also get rid of the noise maps
            # make new nii and save
            img = nb.Nifti1Image(data, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{base}_{modality}.nii')

            # make moma only once
            if modality == 'notnulled':
                print('Generating mask')
                subprocess.run(f'/home/sebastian/abin/3dAutomask -prefix {outFolder}/{base}_moma.nii -peels 3 -dilate 2 {outFolder}/{base}_{modality}.nii',shell=True)

            # make reference image
            reference = np.mean(data[:,:,:,4:6],axis=-1)
            # and save it
            img = nb.Nifti1Image(reference, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{base}_{modality}_reference.nii')

            # separate into individual volumes
            for i in range(data.shape[-1]):
                # overwrite volumes 0,1,2 with volumes 3,4,5
                if i <= 2:
                    vol = data[:,:,:,i+3]
                else:
                    vol = data[:,:,:,i]
                # Save individual volumes
                img = nb.Nifti1Image(vol, header=header, affine=affine)
                nb.save(img, f'{outFolder}/{base}_{modality}_vol{i:03d}.nii')
            # define mask and reference images in 'antspy-style'
            fixed = ants.image_read(f'{outFolder}/{base}_{modality}_reference.nii')
            mask = ants.image_read(f'{outFolder}/{base}_moma.nii')

            # loop over volumes to do the correction
            for i in range(data.shape[-1]):
                moving = ants.image_read(f'{outFolder}/{base}_{modality}_vol{i:03d}.nii')
                mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform = 'Rigid', mask=mask)
                # save transformation matrix for later
                os.system(f"cp {mytx['fwdtransforms'][0]} {outFolder}/motionParameters/{base}/{base}_{modality}_vol{i:03d}.mat")
                # convert transformattion matrix into FSL format
                os.system(f'ConvertTransformFile 3 {outFolder}/motionParameters/{base}/{base}_{modality}_vol{i:03d}.mat {outFolder}/motionParameters/{base}/{base}_{modality}_vol{i:03d}_af.mat --convertToAffineType')
                os.system(f'/home/sebastian/c3d-1.1.0-Linux-x86_64/bin/c3d_affine_tool -ref {outFolder}/{base}_{modality}_reference.nii -src {outFolder}/{base}_{modality}_vol{i:03d}.nii -itk {outFolder}/motionParameters/{base}/{base}_{modality}_vol{i:03d}_af.mat -ras2fsl -o {outFolder}/motionParameters/{base}/{base}_{modality}_vol{i:03d}_FSL.mat -info-full')
                # read parameters
                tmp = fsl.AvScale(all_param=True,mat_file=f'{outFolder}/motionParameters/{base}/{base}_{modality}_vol{i:03d}_FSL.mat');
                tmpReadout = tmp.run();

                # Get the rotations (in rads) and translations (in mm) per volume
                aryTmpMot = list(itertools.chain.from_iterable([tmpReadout.outputs.translations,tmpReadout.outputs.rot_angles]));

                # Save the rotation and translations in lists
                lstsubMot.append(aryTmpMot)
                lstTR_sub.append([int(i)+1 for k in range(6)])
                lstsubMot_Nme.append([f'TX {modality}',f'TY {modality}',f'TZ {modality}',f'RX {modality}',f'RY {modality}',f'RZ {modality}'])
                modalityList.append([modality for k in range(6)])

                clear_output(wait=True)
                # apply transformation
                mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,transformlist=mytx['fwdtransforms'], interpolator='bSpline')
                # save warped image
                ants.image_write(mywarpedimage, f'{outFolder}/{base}_{modality}_vol{i:03d}_warped.nii')

            # assemble images
            newData = np.zeros(data.shape)
            for i in range(data.shape[-1]):
                vol = nb.load(f'{outFolder}/{base}_{modality}_vol{i:03d}_warped.nii').get_fdata()
                newData[:,:,:,i] = vol
            img = nb.Nifti1Image(newData, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{base}_moco_{modality}.nii')
            # remove volumes
            os.system(f'rm {outFolder}/{base}_{modality}_vol*.nii')


        # Make appropriate arrays from lists
        aryCurr = np.array(lstsubMot)
        aryCurr_Ses =  aryCurr.reshape((aryCurr.size,-1))

        aryCurr_TR = np.array(lstTR_sub)
        aryCurr_TR_Ses = aryCurr_TR.reshape((aryCurr_TR.size,-1))

        aryCurr_Nme = np.array(lstsubMot_Nme)
        aryCurr_Nme_Ses = aryCurr_Nme.reshape((aryCurr_Nme.size,-1))

        aryIdx = np.arange(1,len(aryCurr_Nme_Ses)+1)

        aryCurr_mod = np.array(modalityList)
        aryCurr_mod = aryCurr_mod.reshape((aryCurr_mod.size,-1))

        data_dict = {
            'Time/TR': aryCurr_TR_Ses[:,0],
            'Motion_Name': aryCurr_Nme_Ses[:,0],
            'Motion': aryCurr_Ses[:,0],
            'idx':aryIdx,
            'modality': aryCurr_mod[:,0]}

        # Save motion parameters as csv
        pd_ses = pd.DataFrame(data=data_dict)
        pd_ses.to_csv(f'{outFolder}/motionParameters/{base}_motionParameters.csv', index=False)



        os.system(f'/home/sebastian/abin/3dTcat -prefix {outFolder}/{base}_combined.nii  {outFolder}/{base}_moco_notnulled.nii {outFolder}/{base}_moco_nulled.nii -overwrite')

        os.system(f'/home/sebastian/abin/3dTstat -cvarinv -overwrite -prefix {outFolder}/{base}_T1w.nii {outFolder}/{base}_combined.nii')
        os.system(f'/home/sebastian/antsInstallExample-master/install/bin/N4BiasFieldCorrection -d 3 -i {outFolder}/{base}_T1w.nii -o {outFolder}/{base}_T1w_N4Corrected.nii')

    ############################################################################
    ############# Here, the coregistration of multiple runs starts #############
    ############################################################################


    try: # trying to register to first eventStim run

        referenceRun = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-event*run-00*_cbv.nii.gz'))[0]
        refBase = os.path.basename(referenceRun).rsplit('.', 2)[0][:-4]
        runs.remove(referenceRun)

    except: # if not possible, register to first run
        referenceRun = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-*run-00*_cbv.nii.gz'))[0]
        refBase = os.path.basename(referenceRun).rsplit('.', 2)[0][:-4]
        runs.remove(referenceRun)

    # referenceRun = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-block*run-00*_cbv.nii.gz'))[0]
    # refBase = os.path.basename(referenceRun).rsplit('.', 2)[0][:-4]
    # runs.remove(referenceRun)


    fixed = ants.image_read(f'{outFolder}/{refBase}_T1w_N4Corrected.nii')
    mask = ants.image_read(f'{outFolder}/{refBase}_moma.nii')

    for run in runs:
        base = os.path.basename(run).rsplit('.', 2)[0][:-4]
        print(f'Processing run {base}')

        moving = ants.image_read(f'{outFolder}/{base}_T1w_N4Corrected.nii')
        mytx = ants.registration(fixed=fixed, moving=moving, type_of_transform='Rigid',mask=mask)

        os.system(f"cp {mytx['fwdtransforms'][0]} {outFolder}/{base}_T1w_N4Corrected_registered-{refBase}.mat")

        mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving, transformlist=mytx['fwdtransforms'], interpolator='bSpline')

        ants.image_write(mywarpedimage, f'{outFolder}/{base}_T1w_N4Corrected_registered-{refBase}.nii')


        transform1 = f'{outFolder}/{base}_T1w_N4Corrected_registered-{refBase}.mat'

        for start, modality in enumerate(['notnulled', 'nulled']):
            print(modality)


            nii = nb.load(run)
            header = nii.header
            affine = nii.affine
            dataComplete = nii.get_fdata()

            # separate nulled and notnulled data
            data = dataComplete[:,:,:,start:-2:2]

            for i in range(data.shape[-1]):
                if i <= 2:
                    vol = data[:,:,:,i+3]
                else:
                    vol = data[:,:,:,i]

                img = nb.Nifti1Image(vol, header=header, affine=affine)
                nb.save(img, f'{outFolder}/{base}_{modality}_vol{i:03d}.nii')


            for i in range(data.shape[-1]):

                moving = ants.image_read(f'{outFolder}/{base}_{modality}_vol{i:03d}.nii')


                transform2 = f'{outFolder}/motionParameters/{base}/{base}_{modality}_vol{i:03d}.mat'


                mywarpedimage = ants.apply_transforms(fixed=fixed, moving=moving,transformlist=[transform1,transform2], interpolator='bSpline')

                ants.image_write(mywarpedimage, f'{outFolder}/{base}_{modality}_vol{i:03d}.nii')

            newData = np.zeros(data.shape)
    

            for i in range(data.shape[-1]):
                vol = nb.load(f'{outFolder}/{base}_{modality}_vol{i:03d}.nii').get_fdata()
                newData[:,:,:,i] = vol
            img = nb.Nifti1Image(newData, header=header, affine=affine)
            nb.save(img, f'{outFolder}/{base}_moco_{modality}.nii')
            os.system(f'rm {outFolder}/{base}_{modality}_vol*.nii')


        os.system(f'/home/sebastian/abin/3dTcat -prefix {outFolder}/{base}_combined.nii  {outFolder}/{base}_moco_notnulled.nii {outFolder}/{base}_moco_nulled.nii -overwrite')

        os.system(f'/home/sebastian/abin/3dTstat -cvarinv -overwrite -prefix {outFolder}/{base}_T1w.nii {outFolder}/{base}_combined.nii')
        os.system(f'/home/sebastian/antsInstallExample-master/install/bin/N4BiasFieldCorrection -d 3 -i {outFolder}/{base}_T1w.nii -o {outFolder}/{base}_T1w_N4Corrected.nii')
