import subprocess
import glob
import os
import nibabel as nb
import numpy as np
import re
root = '/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti'


def findTR(logfile):
    with open(logfile) as f:
        f = f.readlines()

    triggerTimes = []
    for line in f[1:]:
        if re.findall("Keypress: 5",line):
            triggerTimes.append(float(re.findall("\d+\.\d+", line)[0]))

    triggerTimes[0] = 0

    triggersSubtracted = []
    for n in range(len(triggerTimes)-1):
        triggersSubtracted.append(float(triggerTimes[n+1])-float(triggerTimes[n]))

    meanFirstTriggerDur = np.mean(triggersSubtracted[::2])
    meanSecondTriggerDur = np.mean(triggersSubtracted[1::2])

    # find mean trigger-time
    meanTriggerDur = (meanFirstTriggerDur+meanSecondTriggerDur)/2
    return meanTriggerDur


# for sub in ['sub-03', 'sub-05', 'sub-06']:
for sub in ['sub-09']:
    # for ses in ['ses-001', 'ses-002']:
    for ses in ['ses-001']:
        runs = sorted(glob.glob(f'{root}/{sub}/{ses}/func/{sub}_{ses}_task-*run-00*_cbv.nii.gz'))
        referenceRun = sorted(glob.glob(f'{root}/{sub}/ses-00*/func/{sub}_ses-00*_task-event*run-00*_cbv.nii.gz'))[0]
        outFolder = f'/media/sebastian/Data/EVENTRELATED_PILOT/rawData/Nifti/derivatives/{sub}/{ses}'

        for run in runs[-2:-1]:
            base = os.path.basename(run).rsplit('.', 2)[0][:-4]
            print(f'Processing run {base}')

            for start, modality in enumerate(['notnulled', 'nulled']):
                command = f'/home/sebastian/abin/3dUpsample '
                command += f'-overwrite '
                command += f'-datum short '
                command += f'-prefix {outFolder}/{base}_{modality}_intemp.nii '
                command += f'-n 2 '
                command += f'-input {outFolder}/{base}_moco_{modality}.nii'


                subprocess.call(command, shell=True)

            # nii = nb.load(f'{outFolder}/{base}_notnulled_intemp.nii')
            # header = nii.header
            # affine = nii.affine
            # data = nii.get_fdata()
            #
            # NumVol_notnulled = data.shape[-1]
            # print(f'notnulled timepoints: {NumVol_notnulled}')
            #
            # nii = nb.load(f'{outFolder}/{base}_nulled_intemp.nii')
            # header = nii.header
            # affine = nii.affine
            # data = nii.get_fdata()
            #
            # NumVol_nulled = data.shape[-1]
            # print(f'nulled timepoints: {NumVol_nulled}')
            #
            #
            # if NumVol_nulled < NumVol_notnulled:
            #     newNrVols = NumVol_nulled-1
            # if NumVol_nulled >= NumVol_notnulled:
            #     newNrVols = NumVol_notnulled-2
            #
            # # make new nulled data
            # nii = nb.load(f'{outFolder}/{base}_nulled_intemp.nii')
            # header = nii.header
            # affine = nii.affine
            # data = nii.get_fdata()
            #
            # newData = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
            # for i in range(data.shape[3]):
            #     if i == 0:
            #         newData[:,:,:,i]=data[:,:,:,i]
            #     else:
            #         newData[:,:,:,i]=data[:,:,:,i-1]
            # img = nb.Nifti1Image(newData, header=header, affine=affine)
            # nb.save(img, f'{outFolder}/{base}_nulled_intemp.nii')
            #
            #
            #
            # # make new notnulled data
            # nii = nb.load(f'{outFolder}/{base}_notnulled_intemp.nii')
            # header = nii.header
            # affine = nii.affine
            # data = nii.get_fdata()
            #
            # newData = np.zeros((data.shape[0],data.shape[1],data.shape[2],data.shape[3]))
            # for i in range(data.shape[3]):
            #     newData[:,:,:,i]=data[:,:,:,i]
            # img = nb.Nifti1Image(newData, header=header, affine=affine)
            # nb.save(img, f'{outFolder}/{base}_notnulled_intemp.nii')


            # if NumVol_nulled <= NumVol_notnulled:
            #     print('nulled has less volumes')
            #     newData = np.zeros((data.shape[0],data.shape[1],data.shape[2],NumVol_nulled))
            #     for i in range(NumVol_nulled):
            #         if i == 0:
            #             newData[:,:,:,i]=data[:,:,:,i]
            #         else:
            #             newData[:,:,:,i]=data[:,:,:,i-1]
            #
            #     # Prepare header with new nr of volumes
            #     newHead = header.copy()
            #     newHead['dim'][4]=NumVol_nulled
            #
            #
            #     img = nb.Nifti1Image(newData, header=newHead, affine=affine)
            #     nb.save(img, f'{outFolder}/{base}_nulled_intemp.nii')
            #
            #
            #     nii = nb.load(f'{outFolder}/{base}_notnulled_intemp.nii')
            #     header = nii.header
            #     affine = nii.affine
            #     data = nii.get_fdata()
            #
            #     newData = np.zeros((data.shape[0],data.shape[1],data.shape[2],NumVol_nulled))
            #     for i in range(NumVol_nulled):
            #         if i == 0:
            #             newData[:,:,:,i]=data[:,:,:,i]
            #         else:
            #             newData[:,:,:,i]=data[:,:,:,i-1]
            #
            #     # Prepare header with new nr of volumes
            #     newHead = header.copy()
            #     newHead['dim'][4]=NumVol_nulled
            #
            #
            #     img = nb.Nifti1Image(newData, header=newHead, affine=affine)
            #     nb.save(img, f'{outFolder}/{base}_notnulled_intemp.nii')
            #
            #
            # else:
            #     print('notnulled has less volumes')
            #     newData = np.zeros((data.shape[0],data.shape[1],data.shape[2],NumVol_notnulled))
            #     for i in range(NumVol_notnulled):
            #         if i == 0:
            #             newData[:,:,:,i]=data[:,:,:,i]
            #         else:
            #             newData[:,:,:,i]=data[:,:,:,i-1]
            #     # Prepare header with new nr of volumes
            #     newHead = header.copy()
            #     newHead['dim'][4]=NumVol_notnulled
            #
            #     img = nb.Nifti1Image(newData, header=newHead, affine=affine)
            #     nb.save(img, f'{outFolder}/{base}_nulled_intemp.nii')


            print('correcting TR in header and calculating mean image')
            tr = findTR(f'{root}/derivatives/{sub}/{ses}/events/{base}.log')
            for start, modality in enumerate(['notnulled', 'nulled']):
                subprocess.run(f'/home/sebastian/abin/3drefit -TR {tr} {outFolder}/{base}_{modality}_intemp.nii',shell=True)

                nii = nb.load(f'{outFolder}/{base}_moco_{modality}.nii')
                header = nii.header
                affine = nii.affine
                data = nii.get_fdata()
                mean = np.mean(data,axis=-1)
                img = nb.Nifti1Image(mean, header=header, affine=affine)
                nb.save(img, f'{outFolder}/{base}_{modality}_mean.nii')


            print('running BOCO')
            subprocess.run(f'/home/sebastian/git/laynii/LN_BOCO -Nulled {outFolder}/{base}_nulled_intemp.nii -BOLD {outFolder}/{base}_notnulled_intemp.nii -output {outFolder}/{base}', shell=True)


            subprocess.run(f'fslmaths {outFolder}/{base}_VASO_LN.nii -mul 100 {outFolder}/{base}_VASO.nii.gz -odt short', shell=True)
            subprocess.run(f'fslmaths {outFolder}/{base}_notnulled_intemp.nii  -mul 1 {outFolder}/{base}_BOLD.nii.gz -odt short', shell=True)

            # calculate quality measures
            for modality in ['BOLD', 'VASO']:
                subprocess.run(f'/home/sebastian/git/laynii/LN_SKEW -input {outFolder}/{base}_{modality}.nii.gz', shell=True)
