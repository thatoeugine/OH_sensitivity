import numpy as np
from astropy import constants
from astropy import units as u
from astropy.cosmology import LambdaCDM
from scipy.integrate import quad

cosmos = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def integral(z):
    top = (1 + z) ** 4
    bottom = np.sqrt((cosmos.Om0 * ((1 + z) ** 3)) + cosmos.Ode0)
    return top / bottom


def Total_Volume(zmin, zmax, D=13.5, vOH=1667.35899):
    """
    Function computes a telescopes total volume based on the redshift limits
    Parameters:
    ----------
    zmin and zmax (float): redshift limits
    D (float): the diamter of a single dish in meters
    vOH (float): OH centre frequency in mega-hertz
    """
    A = np.pi * ((1.22 / 2) ** 2)
    B = (
        (constants.c * cosmos.angular_diameter_distance_z1z2(zmin, zmax))
        / ((vOH * u.MHz) * (D * u.m))
    ) ** 2
    C = (constants.c / cosmos.H0).to(u.Mpc)
    return ((A * B * C).to(u.Gpc ** 3)) * quad(integral, zmin, zmax)[0]


def OH_LF_integral(z, Vtot, m=2.2, L_min=10, L_max=10**4.4):
    """
    Function predicts the number of OHMs to be detected at the 3sigma level with MeerKAT
    Parameters:
    ----------
    z (array): redshifts of OH candidates obtained from the catalogue
    Vtot (float): the total available volume
    m (float): the evolution merger rate
    OHLum_min (float): Minimum OH luminosity detectable depending on the sensitivity of observations
    OHLum_max (float): Maximum OH luminosity detectable depending on the sensitivity of observations
    """

    phi_star = 10**-7.36
    L_star = 10**3.59
    alpha = -1.18

    def integrand(L_OH):
        return phi_star * np.log(10) * (L_OH / L_star)**(alpha + 1) * np.exp(-L_OH / L_star)

    result, _ = quad(integrand, L_min, L_max)
    return result * (Vtot) * ((1 + z) ** m),
