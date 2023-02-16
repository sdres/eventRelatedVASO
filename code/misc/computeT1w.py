'''
Computes T1w image in EPI-space from motion-corrected 'nulled' and 'notnulled'
timeseries as acquired with SS-SI VASO.
'''

import numpy as np
import nibabel as nb
from scipy import signal


def computeT1w(nulledFile, notnulledFile, detrend = False):

    '''
    Takes nulled and notnulled files as input and computes T1w image
    in EPI space. Returns array instead of saving a file to allow different
    naming conventions.
    '''

    # Load nulled motion corrected timeseries
    nulledNii = nb.load(nulledFile)
    nulledData = nulledNii.get_fdata()

    # Load notnulled motion corrected timeseries
    notnulledNii = nb.load(notnulledFile)
    notnulledData = notnulledNii.get_fdata()

    # Concatenate nulled and notnulled timeseries
    combined = np.concatenate((notnulledData,nulledData), axis=3)

    if detrend == True:
        # Detrend before std. dev. calculation
        combinedDemean = signal.detrend(combined, axis = 3, type = 'constant')
        combinedDetrend = signal.detrend(combinedDemean, axis = 3, type = 'linear')
        stdDev = np.std(combinedDetrend, axis = 3)
    else:
        stdDev = np.std(combined, axis = 3)

    #Compute mean
    mean = np.mean(combined, axis = 3)
    # Compute variation
    cvar = stdDev/mean
    # Take inverse
    cvarInv = 1/cvar

    return cvarInv
