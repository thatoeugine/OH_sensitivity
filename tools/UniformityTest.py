import numpy as np
import pylab as plt

from tools.VmaxOHLuminosityFunction import Va


def V_Va(V, V_a, OHLum, step=0.5):
    """
    Function computes the average ratio of V/Va for the well-sampled luminosity bins
    Parameters:
    ----------
    V (float): available volume based on the redshift of OH candidate from the catalogue
    V_a (float): available volume based on the maximum redshift of OH candidate that can be observed by Telescope
    OHLum (numpy array): OH luminosities
    step (float):  bin step size, default 0.5  
    """

    start = np.floor(np.min(OHLum) / step) * step
    stop = np.max(OHLum) + step
    bin_edges = np.arange(start, stop, step=step)
    OHLumbins = plt.hist(OHLum, bins=bin_edges)
    OHLumbins_centre = (OHLumbins[1][:-1] + OHLumbins[1][1:]) / 2.0
    plt.close()

    # initialise
    V_Va_bins_mean = np.zeros(len(OHLumbins_centre))
    V_Va_stdbins = np.zeros(len(OHLumbins_centre))
    NOHperbin = np.zeros(len(OHLumbins_centre))

    for b in range(len(OHLumbins_centre)):
        # mask of OH luminosity in this Luminosity bin
        maskb = np.argwhere((OHLum >= OHLumbins[1][b]) & (OHLum < OHLumbins[1][b + 1]))
        maskb = np.ravel(maskb)

        NOHperbin[b] = maskb.sum()  # total number of OH sources in this luminosity bin

        if NOHperbin[b] == 0:
            V_Va_bins_mean[b], V_Va_stdbins[b] = (
                0,
                0,
            )  # if no sources in bin , set to zero
        else:

            V_Va_bins_mean[b] = np.nanmean((V[maskb].value) / (V_a[maskb].value))
            V_Va_stdbins[b] = np.nanstd((V[maskb].value) / (V_a[maskb].value))

    return OHLumbins_centre, V_Va_bins_mean, V_Va_stdbins
