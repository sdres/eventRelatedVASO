import pydicom

folder = '/Users/sebastiandresbach/data/eventRelatedVASO/DICOM/sub-14/ses-001'
file = 'SEBDRE2022031403.MR.AMANDA_KAAS_SEBDRE.0003.0211.2022.03.14.17.33.25.258425.167642064.IMA'

ds = pydicom.dcmread(f'{folder}/{file}')
