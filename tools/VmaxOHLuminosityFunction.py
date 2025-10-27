import numpy as np
import pylab as plt
from astropy import constants
from astropy import units as u
from astropy.cosmology import LambdaCDM
from scipy.integrate import quad
from scipy.optimize import curve_fit, fsolve, least_squares

from tools.SensitivityLimit import f_lim

cosmos = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def OHflux(OHLum, z, V=150, vOH=1667):
    """
    OH Luminosity to OH flux density conversion
    Parameters:
    ----------
    OHLum(numpy array): OH luminosities
    z (numpy array): redshift of OH candidate from the catalogue
    V (float): Average FWHM of OH source at z = 0 in km/s
    vOH (float): OH centre frequency in mega-hertz
    """
    OHLum = 10 ** (OHLum)
    a = (OHLum * u.Lsun) * (constants.c) * (1 + z)
    b = (
        (4 * np.pi)
        * (cosmos.luminosity_distance(z) ** 2)
        * (V * u.km / u.s)
        * (vOH * u.MHz)
    )
    return (a / b).to(u.Jy)


# ================ computing the maximum redshift =======================#
"""
The functions below computes the maximum detectable redshift of OH candidates 
using equation 8 from Darling (2002).
"""


def integral(z):
    """
    LHS which supposed to integrates from 0 up to zmax
    """
    return (((1 + z) ** 3) * cosmos.Om0 + cosmos.Ode0) ** (-0.5)


def zmax_solve(zmax, flux, z, rms=155e-6 * u.Jy):
    a = (cosmos.H0 * cosmos.luminosity_distance(z)) / (
        constants.c.to(u.km / u.s) * np.sqrt(1 + z)
    )
    b = (((flux * u.Jy) / (rms)) ** (0.5)) * ((1 + zmax) ** (-0.5))

    return quad(integral, 0, zmax) - (a * b).value


# =========================================================================


def Va(z_max, z_min=0, AreaCoverage=940.7, AreaCoverage_units="sqdeg"):
    """
    Function computes the maximum available Volume of the candidate OHM
    Parameters:
    ----------
    AreaCoverage (float): The area coverage of the survey field (from the HELP catalogue) in sqdeg or sr
    z_min (numpy array): minimum redshift of OH candidate that can be observed by Telescope
    z_max (numpy array): maximum redshift of OH candidate that can be observed by Telescope
    """
    if AreaCoverage_units == "sqdeg":
        SolidAngle = AreaCoverage / ((180 / np.pi) ** 2)  # sr
    else:
        SolidAngle = AreaCoverage  # in sr
    DL_min = cosmos.luminosity_distance(z_min)
    DL_max = cosmos.luminosity_distance(z_max)
    V = (SolidAngle / 3) * ((DL_max / (1 + z_max)) ** 3 - (DL_min / (1 + z_min)) ** 3)
    return V


def LuminosityFunction(OHLum, V_a, step=0.5):
    """
    Function computes the Luminosity function of OHM
    Parameters:
    ----------
    OHLum (numpy array): OH luminosities
    V_a (numpy array): The maximium available volume for OHM
    step (float): bin step size, default is 0.5
    """

    start = np.floor(np.min(OHLum) / step) * step
    stop = np.max(OHLum) + step
    bin_edges = np.arange(start, stop, step=step)
    OHLumbins = plt.hist(OHLum, bins=bin_edges)
    OHLumbins_centre = (OHLumbins[1][:-1] + OHLumbins[1][1:]) / 2.0
    plt.close()

    # initialise
    OHLumFunbins_mean = np.zeros(len(OHLumbins_centre))
    stdbins = np.zeros(len(OHLumbins_centre))
    NOHLumperbin = np.zeros(len(OHLumbins_centre))

    for b in range(len(OHLumbins_centre)):
        # mask of OH luminosity in this Luminosity bin
        maskb = np.argwhere((OHLum >= OHLumbins[1][b]) & (OHLum < OHLumbins[1][b + 1]))
        maskb = np.ravel(maskb)
        NOHLumperbin[b] = maskb.sum()

        OHLumFunbins_mean[b] = np.nanmean(
            (1 / OHLum[maskb]) * np.sum(1 / V_a[maskb].value)
        )
        stdbins[b] = np.nanstd(
            (1 / OHLum[maskb]) * (np.sum(1 / V_a[maskb].value) ** 0.5)
        ) / np.sqrt(NOHLumperbin[b])
    return OHLumbins_centre, OHLumFunbins_mean, stdbins, NOHLumperbin
