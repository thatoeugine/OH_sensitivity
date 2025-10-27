import numpy as np
from astropy import constants
from astropy import units as u
from astropy.cosmology import LambdaCDM

cosmos = LambdaCDM(H0=70, Om0=0.3, Ode0=0.7)


def f_lim(redshift, SEFD=500, Ncorr=1,
          Npol=2, Nant=58, time_int=8,
          bandwidth=133, sigma_level=1,
          restV=150, restfreq=1667):
    """
    Function calculates MeerKAT's sensitivity at a sigma level based on the SEFD and redshift
    Parameters:
    ----------
    redshift (float): source of interest redshift
    SEFD (int): SEFD of the MeerKAT telescope
    Ncorr (int): Number of correlators
    Npol (int): Number of polarisations
    Nant (int): Number of antennas
    time_int (int): Integration time in hours
    bandwidth (float): Bandwidth in kilo-Hertz
    restV (float):  rest-frame velocity width in km/s
    restfreq (float): rest-frame frequency in MHz
    _________
    returns the 3sigma sensitivity in Lsun and Jy
    """

    Smin = (SEFD * u.Jy) / (
        Ncorr
        * np.sqrt(Npol * Nant * (Nant - 1) * (time_int * u.hour) * (bandwidth * u.kHz))
    )  # Telescope's sensitivity based on the SEFD
    DL = cosmos.luminosity_distance(redshift)

    Lmin = (4 * np.pi) * (DL ** 2) * (sigma_level * Smin.to(u.Jy)) *\
        ((restfreq * u.MHz) * (restV * u.km/u.s) /
         (constants.c.to(u.km/u.s)*(1 + redshift)))  # Luminosity Sensitivity

    return np.log10(Lmin.to(u.Lsun).value), sigma_level * Smin.to(u.Jy)


def ChannelVelocitywidth(f_obs, Totalbandwidth=544, correlator='4k'):
    """
    Function gives output the channel width and velocity width 
    of an observed line
    """

    if correlator == '4k':
        channel_width = (Totalbandwidth/4096)*u.MHz
    if correlator == '32k':
        channel_width = (Totalbandwidth/32768)*u.MHz
    f_obs_ = f_obs*u.MHz
    velocity_width = (constants.c*(channel_width.to(u.Hz)))/(f_obs_.to(u.Hz))

    return channel_width.to(u.kHz), velocity_width.to(u.km/u.s)


def Observed_freq(z, restfreq=1667):
    """
    Function converts redshift to observed frequency
    """
    return restfreq*u.MHz/(z + 1)


def OHflux(OHLum, z, bandwidth=133):
    """
    OH Luminosity to OH flux density conversion
    Parameters:
    ----------
    OHLum(numpy array): OH luminosities
    z (numpy array): redshift of OH candidate from the catalogue
    bandwidth (float): Telescope's bandwidth in kilo-hertz
    """
    OHLum = 10 ** (OHLum)
    a = OHLum * u.Lsun
    b = (
        (4 * np.pi)
        * (cosmos.luminosity_distance(z) ** 2)
        * (bandwidth * u.kHz)
    )
    return (a / b).to(u.Jy)


def OHpeakflux(OHLum, z, obsfreq=1667):

    OHLum = 10 ** (OHLum) * u.Lsun
    DL = cosmos.luminosity_distance(z).to(u.Gpc)
    a = 1.0234 * obsfreq * (DL ** 2)

    return OHLum/a


"""
Fline = OHpeakflux(OHLum = sorted_df['logOHLum_solar'].values, z = sorted_df['zspec'].values,
            obsfreq= Observed_freq(sorted_df['zspec'].values, restfreq = 1667.35899))
"""
