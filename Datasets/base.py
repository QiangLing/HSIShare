"""
Base IO code for all HSI datasets.
We use the same IO as sklearn.
"""
import numpy as np
import scipy.io as spio
from sklearn.datasets.base import Bunch
from os.path import dirname, join
import itertools

__all__ = [ 'load_SHARE2012']


def load_SHARE2012(return_X_y=True, return_TarDict=False, return_Unmix=False, ROI='AVON_AM'):
    """Load and return the Viareggio 2013 dataset (Anomaly Detection or Target Detection).

    Parameters
    ----------
    return_X_y : boolean, default=True.
        If True, returns ``(HSI, GroTru, GoodBands)`` instead of a Bunch object.
    return_TarDict:
    return_Unmix:
    ROI : string, region of interest, default='D1_F12_H1'.
        It can be 'D1_F12_H1', 'D1_F12_H2', and 'D2_F22_H2'

        .. versionadded:: 0.11

    Returns
    -------
    data : Bunch
        Dictionary-like object, the interesting attributes are:
        'HSI', the HSI dataset, 'GroTru', the ground truth map,
        'GoodBands', the good bands of the dataset, 511 bands are used.

    (HSI, GroTru, GoodBands) : tuple if ``return_X_y`` is True

        .. versionadded:: 0.11

    Examples
    --------
    >>> from LQ.Datasets import load_Viareggio2013
    >>> data = load_Viareggio2013(return_X_y=False)
    >>> list(data)
    ['HSI', 'GroTru', 'GoodBands']
    """
    module_path = dirname(__file__)
    if ROI == 'AVON_AM':
        FileName = 'AVON_AM1.mat'   # AVON_AM for target detection, AVON_AM1 for anomaly detection
    elif ROI == 'D1_F12_H2':
        # TODO:
        FileName = 'D1_F12_H2_Cropped_NW.mat'
    elif ROI == 'D2_F22_H2':
        # TODO:
        FileName = 'D2_F22_H2_Cropped_NW.mat'
    elif ROI == 'No Target':
        FileName = 'AVON_AMNoTar.mat'
    elif ROI == 'Unmix':
        FileName = 'AVON_AMUnmix1.mat'  # AVON_AMUnmix
    data = spio.loadmat(join(module_path, 'HSIDatasets', 'SHARE 2012', 'AVON', FileName), mat_dtype=True)
    HSI = data['HSI']
    GroTru = data['GroTru']
    if return_Unmix:
        TarDict = data['TarDict']
        GroTruUnmix1_1 = data['GroTruUnmix1_1']
        GroTruUnmix1_2 = data['GroTruUnmix1_2']
        GroTruUnmix1_3 = data['GroTruUnmix1_3']
        GroTruUnmix2_1 = data['GroTruUnmix2_1']
        GroTruUnmix2_2 = data['GroTruUnmix2_2']
        GroTruUnmix2_3 = data['GroTruUnmix2_3']

    if return_TarDict:
        TarDict = data['TarDict']
    GoodBands = np.r_[:360]
    if return_X_y:
        if return_Unmix:
            return HSI, GroTru, TarDict, GoodBands, GroTruUnmix1_1, GroTruUnmix1_2, GroTruUnmix1_3, GroTruUnmix2_1, GroTruUnmix2_2, GroTruUnmix2_3
        elif return_TarDict:
            return HSI, GroTru, TarDict, GoodBands
        else:
            return HSI, GroTru, GoodBands

    if return_Unmix:
        return Bunch(HSI=HSI, GroTru=GroTru, TarDict=TarDict, GoodBands=GoodBands,
                     GroTruUnmix1_1=GroTruUnmix1_1, GroTruUnmix1_2=GroTruUnmix1_2,
                     GroTruUnmix1_3=GroTruUnmix1_3, GroTruUnmix2_1=GroTruUnmix2_1,
                     GroTruUnmix2_2=GroTruUnmix2_2, GroTruUnmix2_3=GroTruUnmix2_3)
    elif return_TarDict:
        return Bunch(HSI=HSI, GroTru=GroTru, TarDict=TarDict, GoodBands=GoodBands)
    else:
        return Bunch(HSI=HSI, GroTru=GroTru, GoodBands=GoodBands)


