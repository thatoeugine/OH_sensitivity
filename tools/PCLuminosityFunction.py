"""
P&C Luminosity function
---------
Author:Thato Manamela  
Version: 2022
-------------------------------------------------------------------------------------------------
This code Computes the Luminosity function using the Page & Carrera's method.
 Ref: Page, M. J., & Carrera, F. J. 2000, MNRAS, 311, 433. 
-------------------------------------------------------------------------------------------------
"""


import numpy as np
import pylab as plt
from astropy import units as u
from astropy.cosmology import LambdaCDM
from astropy.stats import poisson_conf_interval as pci
from scipy.integrate import dblquad

cosmos = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)  # Cosmology


class LuminosityFunction(object):
    """
    Computes the Luminosity function using the Page & Carrera's method.
    Ref: Page, M. J., & Carrera, F. J. 2000, MNRAS, 311, 433
    Parameters:
    ----------
    Lum (numpy array): luminosities/ Absolute magnitudes (Mi) objects
    z (numpy array): redshifts of objects
    zbin (list): redshift bin
    step (float): bin step size, default is 0.5
    AreaCoverage (float): Surveys area coverage sr/sqdeg
    AreaCoverage_units (str): Surveys area coverage units sr/sqdeg
    """

    def __init__(
        self,
        Lum,
        z,
        zbin=[0.30, 0.55],
        step=0.5,
        AreaCoverage=940.7,
        AreaCoverage_units="sqdeg",
    ):
        self.Lum = Lum
        self.z = z
        self.zbin = zbin
        self.step = step
        self.AreaCoverage = AreaCoverage
        self.AreaCoverage_units = AreaCoverage_units

    def dVdz(self, Lum, z):
        """
        Function Calculates the Differential comoving volume using Hoggs Hogg (1999) expression
        Params:
        ------
        Lum (numpy array): luminosities/ Absolute magnitudes (Mi) objects
        z (numpy array): redshifts of objects
        """
        if self.AreaCoverage_units == "sqdeg":
            SolidAngle = self.AreaCoverage / ((180 / np.pi) ** 2)  # sr
        else:
            SolidAngle = self.AreaCoverage  # in sr
        dvdz = cosmos.differential_comoving_volume(z) * (SolidAngle * u.sr)
        return dvdz.value

    def luminosityfunction(self):
        """
        Function computes the Luminosity function using the Page & Carrera's method.
        Ref: Page, M. J., & Carrera, F. J. 2000, MNRAS, 311, 433
        """
        # Binning the Luminosities
        start = np.floor(np.min(self.Lum) / self.step) * self.step
        stop = np.max(self.Lum) + self.step
        bin_edges = np.arange(start, stop, step=self.step)
        Lumbins = plt.hist(self.Lum, bins=bin_edges)
        Lumbins_centre = (Lumbins[1][:-1] + Lumbins[1][1:]) / 2.0
        plt.close()

        # initialise
        LumFunbins = np.zeros(Lumbins_centre.size)
        NoLumperbin = np.zeros(Lumbins_centre.size)

        for b in range(Lumbins_centre.size):
            # masking of luminosities
            maskb = np.argwhere(
                (self.Lum >= Lumbins[1][b])
                & (self.Lum < Lumbins[1][b + 1])
                & (self.z[b] >= self.zbin[0])
                & (self.z[b + 1] < self.zbin[1])
            )
            maskb = np.ravel(maskb)
            NoLumperbin[b] = maskb.size

            if NoLumperbin[b] == 0:
                LumFunbins[b] = 0  # if no sources, set to zero

            else:
                LumFunbins[b] = np.nanmedian(
                    NoLumperbin[b]
                    / dblquad(
                        self.dVdz,
                        self.zbin[0],
                        self.zbin[1],
                        lambda z: self.Lum[maskb].min(),
                        lambda z: self.Lum[maskb].max(),
                    )[0]
                )

        """
        Calculate errorbars on our binned LF.  These have been estimated
        using Equations 1 and 2 of Gehrels 1986 (ApJ 303 336), as
        implemented in astropy.stats.poisson_conf_interval.  The
        interval='frequentist-confidence' option to that astropy function is
        exactly equal to the Gehrels formulas. (also see Kulkarni et al. 2019, MNRAS, 488, 1035)           
        """

        nums = NoLumperbin
        lgphi = LumFunbins
        nlims = pci(nums, interval="frequentist-confidence")
        nlims *= lgphi / nums
        lgphi = np.log10(lgphi)
        uperr = np.log10(nlims[1]) - lgphi
        downerr = lgphi - np.log10(nlims[0])
        stdbins = np.vstack((uperr, downerr))

        return Lumbins_centre, lgphi, stdbins, NoLumperbin

