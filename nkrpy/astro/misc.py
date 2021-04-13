"""Astronomy functions.

def Keplerian_Rotation(mass, velocity, Distance, inclination):
    radii_return =  np.sin(inclination)**2*const.G.value*mass*
        const.M_sun.value/(velocity*1000)/(velocity*1000)/
        (Distance*u.pc.to(u.m))*u.rad.to(u.arcsec)
    #All the positive radii.
    radii_positive = radii_return[velocity < 0]
    #We also have some negative radii, so thats why we have to do this.
    radii_negative = -1*radii_return[velocity > 0]
    return radii_positive, radii_negative
"""

# internal modules

# external modules
import numpy as np

# from scipy.optimize import curve_fit

# relative modules
from ..misc.constants import h, c, kb, msun, jy, mh
from .dustmodels import kappa
from nkrpy import Unit as nc__unit  # noqa
from nkrpy import constants

# global attributes
__all__ = ("dustmass", "planck_nu", "planck_wav", "ecc", "construct_lam", "freq_vel", "vel_freq")
__doc__ = """."""
__filename__ = __file__.split("/")[-1].strip(".py")
__path__ = __file__.strip(".py").strip(__filename__)


# find a better spot for WCS. Probably in io, or need a coordinates library
# also there is misc trash in nkrpy.io.files.functions regarding header generation


def freq_vel(freq: np.ndarray, restfreq: float):
    return (restfreq - freq) / restfreq * constants.c


def vel_freq(vel: np.ndarray, restfreq: float):
    return restfreq - (vel / constants.c * restfreq)


def construct_lam(lammin, lammax, Res=None, dlam=None):
    """Construct a wavelength grid by specifying either a resolving power (`Res`)
    or a bandwidth (`dlam`)
    Parameters
    ----------
    lammin : float
        Minimum wavelength [microns]
    lammax : float
        Maximum wavelength [microns]
    Res : float, optional
        Resolving power (lambda / delta-lambda)
    dlam : float, optional
        Spectral element width for evenly spaced grid [microns]
    Returns
    -------
    lam : float or array-like
        Wavelength [um]
    dlam : float or array-like
        Spectral element width [um]
    """

    # Keyword catching logic
    goR = False
    goL = False
    if ((Res is None) and (dlam is None)) or (Res is not None) and (dlam is not None):
        print("Error in construct_lam: Must specify either Res or dlam, but not both")
    elif Res is not None:
        goR = True
    elif dlam is not None:
        goL = True
    else:
        print("Error in construct_lam: Should not enter this else statment! :)")
        return None, None

    # If Res is provided, generate equal resolving power wavelength grid
    if goR:

        # Set wavelength grid
        dlam0 = lammin/Res
        dlam1 = lammax/Res
        lam  = lammin #in [um]
        Nlam = 1
        while (lam < lammax + dlam1):
            lam  = lam + lam/Res
            Nlam = Nlam +1
        lam    = np.zeros(Nlam)
        lam[0] = lammin
        for j in range(1,Nlam):
            lam[j] = lam[j-1] + lam[j-1]/Res
        Nlam = len(lam)
        dlam = np.zeros(Nlam) #grid widths (um)

        # Set wavelength widths
        for j in range(1,Nlam-1):
            dlam[j] = 0.5*(lam[j+1]+lam[j]) - 0.5*(lam[j-1]+lam[j])

        #Set edges to be same as neighbor
        dlam[0] = dlam0#dlam[1]
        dlam[Nlam-1] = dlam1#dlam[Nlam-2]

        lam = lam[:-1]
        dlam = dlam[:-1]

    # If dlam is provided, generate evenly spaced grid
    if goL:
        lam = np.arange(lammin, lammax+dlam, dlam)
        dlam = dlam + np.zeros_like(lam)

    return lam, dlam


def ecc(a, b):
    """Determine eccentricity given semi-major and semi-minor."""
    return (1. - ((a ** 2) / (b ** 2))) ** 0.5


def inverse_planck_wav(radiance=None, val=None, unit=None):
    """Inverse Plank Function in wavelength."""
    wav = nc__unit(baseunit=unit, convunit='cm', vals=val)
    a = 2.0 * h * c ** 2
    b = (h * c / (wav * kb))
    temp = b / np.log(a / (wav ** 5 * radiance) + 1)
    # returns in units of K
    return temp


def planck_wav(temp=None, val=None, unit=None):
    """Plank Function in wavelength."""
    wav = nc__unit(baseunit=unit, convunit='cm', vals=val)
    a = 2.0 * h * c ** 2
    b = (h * c / (wav * kb * temp))
    intensity = a / ((wav ** 5) * (np.exp(b) - 1.0))
    # returns in units of erg / s / cm^2 / steradian / cm
    return intensity


def planck_nu(temp=None, val=None, nu_unit=None):
    """Plank Function in frequency."""

    nu = nc__unit(baseunit=nu_unit, convunit='hz', vals=val).get_vals()[0]
    a = 2.0 * h / c ** 2
    b = (h * nu / (kb * temp))
    intensity = a * (nu ** 3) / (np.exp(b) - 1.0)
    # returns in units of erg / s / cm^2 / steradian / hz
    return intensity


def dustmass(dist=100, dist_unit="pc", val=0.1,
             val_unit="cm", flux=0, temp=20,
             model_name="oh1994", beta=1.7, gas_density=0, opacity=None):
    """Calculate dust mass.

    @param dist
    dist_unit
    wavelength
    wavelength_unit
    flux
    temp
    model
    beta
    opacity
    Assuming temp in Kelvin, flux in Janskys
    """
    dist = nc__unit(baseunit=dist_unit, convunit='cm', vals=dist).get_vals()[0]  # to match the opacity units
    wav = nc__unit(baseunit=val_unit, convunit='microns', vals=val)  # to search opcaity models
    intensity = np.ravel(planck_nu(temp, wav('hz'), "hz"))[0] * 1.E26  # noqa
    if not opacity:
        opacity = kappa(wav.get_vals()[0],
                        model_name=model_name,
                        density=gas_density,
                        beta=beta)  # cm^2/g
    if not isinstance(opacity, (tuple, list, np.ndarray)):
        opacity = [opacity]
    print(opacity, type(opacity))
    _ret = []
    for x in opacity:
        print(type(x), type(dist), type(flux), type(intensity))
        _tmp = (dist ** 2 * flux / x / intensity / msun)  # noqa in msun units (no ISM assump.)
        _ret.append([x, _tmp])
    return np.array(_ret)


def true_emissive_mass(flux,
                       freq,
                       lam,
                       distance,
                       Tex,
                       pixelarea):
    """."""
    # apply Goldsmith and Langer 1999 to convert each pixel into a mass
    # beam area
    # The below math does this:
    #   Jy*km/s / beam to Jy km/s / area to K*km/s / area
    #  to num_mol/area to num_mol_h2/area to num_mol_h2/pix

    c = c * 1E8  # A/s
    h = h * 1E-7  # SI
    kb = kb * 1E-7  # SI
    theta = 0.27 * 0.19  # arcsec^2
    theta = theta * np.pi * (1./(60. * 60.)) ** 2 *\
        (np.pi / 180.) ** 2  # square degrees
    jybm2jy = (lam / 10) ** 2 * jy / (2. * theta * kb) * 4. * 0.693
    w = flux * jybm2jy  # Jy*km/s / beam to Jy*km/s /area to K km/s / area
    # print("conv: ",jybm2jy)

    # Partition Function
    a = 2.321e-06    # s^-1
    B = 56179.99E6  # Hz
    z = kb * Tex / (h * B)
    # print("Partition: " + str(z))
    j = 3
    g = 2 * j + 1
    # mu = 0.11034
    E = h * B * (j * (j + 1))
    # print(E/k)

    n = (8. * np.pi * kb * freq ** 2 * w) / (h * c ** 3 * a)

    # lnN_tot = np.log(n/g) + np.log(z) + (E / (kb * Tex))
    # N_tot=np.exp(lnN_tot)
    N_tot = n * z / g * np.exp(E / (kb * Tex)) * 1.0e5
    # print("N_tot: ",N_tot)
    N_mol = N_tot

    abun_ratio_c18o_c17o = float(4.16)
    abun_ratio_c18o_h2 = float(1.7E-7)
    # abun_ratio_c17o_co = float(1. / 2000.)
    # abun_ratio_co_h2 = float(10 ** -4)
    abun_ratio_c17o_h2 = 1. / (abun_ratio_c18o_h2 / abun_ratio_c18o_c17o)

    N_mol_h2 = N_mol * abun_ratio_c17o_h2
    # print("N_mol_h2: ", N_mol_h2)

    mol_mass = 2.71 * mh * pixelarea
    # mean mol mass / avogadro #3.34E-24  # grams per H2 molecule
    # print("pixelarea: ",pixelarea)
    Mass_h2 = N_mol_h2 * mol_mass
    return Mass_h2  # grams


def equivalent_width(spectra, blf, xspec0, xspec1, fit="gauss",
                     params=[1, 1, 1]):
    """Compute spectral line eq. width.

    Finds equivalent width of line
    spectra is the full 2d array (lam, flux)
    blf is the baseline function
    xspec0 (xspec1) is the start(end) of the spectral feature


    # PROBABLY EASIER TO JUST ASSUME GAUSSIAN OR SIMPLE SUM


    def gaussc(x, A, mu, sig, C):
        return A*np.exp(-(x-mu)**2/2./sig**2) + C

    from scipy.optimize import curve_fit

    featx, featy = lam[featurei], flux[featurei]
    expected1=[1.3, 2.165, 0.1, 1.]
    params1, cov1=curve_fit(gaussc, featx, featy, expected1)
    sigma=params1[-2]

    # I(w) = cont + core * exp (-0.5*((w-center)/sigma)**2)
    # sigma = dw / 2 / sqrt (2 * ln (core/Iw))
    # fwhm = 2.355 * sigma = 2(2ln2)^0.5 * sigma
    # flux = core * sigma * sqrt (2*pi)
    # eq. width = abs (flux) / cont

    fwhm = 2. * (2. * np.log(2.))**0.5 * sigma
    core, center, nsig, cont = params1
    flx = core * sigma * np.sqrt (2.*np.pi)
    eqwidth = abs(flx) / cont
    print(eqwidth)
    print(fwhm)


    specfeatxi = np.array(between(spectra, xspec0, xspec1))[:, 0]
    specfeatxv = np.array(between(spectra, xspec0, xspec1))[:, 1]

    if fit == "gauss":
        _params2, _cov2 = curve_fit(gauss, specfeatxv, spectra[specfeatxi, 1],
                                    *params)
        _sigma2 = np.sqrt(np.diag(_cov2))
        function = gauss(specfeatxv, *_expected2)
        return
    elif fit == "ndgauss":
        return

    """
    pass


# end of file
