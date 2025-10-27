import numpy as np
from astropy import units as u

def Briggs_OHLFVerification(OHlum, fm = 1):
    """
    Function computes Briggs (1998) OH luminosity function
    """
    C = 5.8e-3 * u.Mpc ** -3
    OHLum_star = 0.10 * u.L_sun
    alpha = 0.49
    beta = 1.81

    a = 0.231 * C * fm
    b = (alpha + beta) / 2
    c = (OHlum* u.L_sun / OHLum_star) ** (-((alpha + beta) / 2))

    return a * b * c

