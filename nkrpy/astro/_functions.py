"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import numpy as np
from scipy.ndimage.interpolation import rotate, shift
from scipy.integrate import quad

# relative modules
from .._types import WCSClass
from ..misc.errors import ArgumentError
from ..misc import constants as n_constants
(_h, _c, _kb, _msun, _jy, _mh) = map(lambda x: getattr(n_constants, x), ('h', 'c', 'kb', 'msun', 'jy', 'mh'))
from .._unit import Unit as nc__unit  # noqa
from ..io import fits as nkrpy_fits
from ..misc.decorators import validate


# global attributes
__all__ = ["blackbody_hz", "blackbody_cm", "opticaldepth_attenuation", "opacity_attenuation", "ecc", "construct_lam", "freq_vel", "vel_freq", 'k_2_jy', 'jy_2_k', 'convert_file', 'select_from_position_axis', 'rms3d', 'rms2d','binning', 'center_image', 'sum_image', 'select_ellipse', 'select_rectangle', 'select_conic_section','select_circle','select_circular_annulus','select_elliptical_annulus','remove_padding3d', 'vel_2_freq', 'freq_2_vel']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def vel_2_freq(center_vel=0, rel_vel=0, restfreq=1):
    # vel in cm
    # restfreq in hz
    return  restfreq - (center_vel + rel_vel) / constants.c * restfreq

def freq_2_vel(restfreq=1,freq=1,  center_vel=0):
    # freq in hz
    # vvel in cm
    return  (restfreq - freq) / restfreq * constants.c + center_vel


def select_ellipse(datashape, xcen, ycen, sma, smi, pa=0, inverse: bool = False):
    """
    pa in radians.
    """
    # print(datashapes, xcen, ycen, sma, smi, pa)
    pa = pa % 360.
    rad = pa * np.pi / 180
    x, y = np.indices(datashape)
    y = y[::-1]
    xp = (x - xcen) * np.cos(rad) + (y - ycen) * np.sin(rad)
    yp = -1. * (x - xcen) * np.sin(rad) + (y - ycen) * np.cos(rad)
    mask = ((xp / sma) ** 2 + (yp / smi) ** 2) <= 1
    return mask if not inverse else ~mask

def select_rectangle(datashape, xcen: float, ycen: float, xlen: float, ylen: float, pa: float = 0, inverse: bool = False):
    """
    pa in degrees.
    """
    pa = pa % 360.
    rad = -pa * np.pi / 180.
    cosa = np.cos(rad)
    sina = np.sin(rad)
    y, x = np.indices(datashape)
    y = y.astype(float)
    x = x.astype(float)
    xp = x - xcen
    yp = y - ycen
    xn = (xp * cosa) + (yp * sina)
    yn = (yp*cosa) - (xp*sina)
    mask = ((np.abs(xn) <= abs(xlen)) * (np.abs(yn) <= abs(ylen)))
    return mask if not inverse else ~mask


def select_conic_section(datashape, xcen, ycen, r1, r2, pa, angle, inverse: bool = False):
    y,x = np.indices(datashape)
    y = y[::-1]
    rsqrd = (x - xcen) ** 2 + (y - ycen) **2
    r = np.sqrt(rsqrd)
    theta = (-np.arctan2((y - ycen), (x - xcen)) - np.pi/2)
    rad1 = -np.pi / 180 * (pa - angle / 2)
    rad2 = -np.pi / 180 * (pa + angle / 2)
    rad1, rad2 = (rad1, rad2) if rad1 > rad2 else (rad2, rad1)
    r1, r2 = (r1, r2) if r1 > r2 else (r2, r1)
    mask = (theta < rad1) * (theta > rad2) * (r < r1) * (r > r2)
    return mask if not inverse else ~mask


def select_circle(datashape, xcen, ycen, radius, inverse: bool = False):
    y,x = np.indices(datashape)
    y = y[::-1]
    rsqrd = (x - xcen) ** 2 + (y - ycen) **2
    r = np.sqrt(rsqrd)
    mask = (r < radius)
    return mask if not inverse else ~mask


def select_circular_annulus(datashape, xcen, ycen, r1, r2, inverse: bool = False):
    outer = select_circle(datashape, xcen, ycen, radius=r1)
    inner = select_circle(datashape, xcen, ycen, radius=r2)
    outer *= ~inner
    return outer if not inverse else ~outer


def select_elliptical_annulus(datashape, xcen, ycen, a1, b1, pa1, a2, b2, pa2, inverse: bool = False):
    ell1 = select_elliptical_annulus(datashape, xcen, ycen, a1, b1, pa1)
    ell2 = select_elliptical_annulus(datashape, xcen, ycen, a2, b2, pa2)
    return ell1 - ell2 if not inverse else ~ell1


def find_padding2d(data):
    upper_rows = 0
    lower_rows = 0
    left_cols = 0
    right_cols = 0
    cont = 1
    for row in range(data.shape[0] // 2):
        if np.isnan(data[row, :, ...]).all():
            lower_rows += 1
        if np.isnan(data[-row, :, ...]).all():
            upper_rows += 1
    for col in range(data.shape[1] // 2):
        if np.isnan(data[:, col, ...]).all():
            left_cols += 1
        if np.isnan(data[:, -col, ...]).all():
            right_cols += 1
    return (lower_rows, left_cols), (upper_rows, right_cols)


def remove_padding3d(data):
    # assume padding is None
    # returns pad length
    valid_cols = (~np.isnan(data)).sum(axis=-1).sum(axis=0).astype(bool)
    valid_rows = (~np.isnan(data)).sum(axis=-1).sum(axis=1).astype(bool)
    return data[valid_rows, ...][:, valid_cols, ...], (valid_rows, valid_cols)


def binning(x, y, windowsize=0.1):
    newx, newy = [], []
    for xi in x:
        mask = np.abs(x - xi) < windowsize
        xmed = np.median(x[mask])
        if xmed in newx:
            continue
        newx.append(xmed)
        newy.append(np.median(y[mask]))
    return np.array(newx), np.array(newy)


def center_image(image, ra, dec, wcs, **kwargs):
    xcen = wcs(ra, return_type='pix', axis=wcs.axis1['type'])
    ycen = wcs(dec, return_type='pix', axis=wcs.axis2['type'])
    imcen = list(map(lambda x: x / 2, image.shape[1:]))
    center_shift = list(map(int, [imcen[0] - ycen, imcen[1] - xcen]))
    shifting = [0]
    shifting.extend(center_shift)
    shifted_image = shift(input=image, shift=shifting, **kwargs)
    return shifted_image


def sum_image(image, width: int = -1, axis: int = -1):
    """Sum along the first axis given a width."""
    image = image if axis == 0 else image.T
    if width is None or width <= 0 or width == np.inf or width == np.nan:
        width = int(image.shape[0] / 2) - 1
    width = width if width % 2 == 1 else width + 1
    cen = int(image.shape[0] / 2)
    summed_image = np.nansum(image[cen - width:cen + width, ...],
                          axis=0)
    return summed_image


def rms3d(cube: np.ndarray):
    """Calulcate the rms of a cube."""
    img = np.nanmedian(cube, axis=-1)
    return rms2d(img)

def rms2d(image: np.ndarray):
    """Calulcate the rms of a cube."""
    cont = np.percentile(np.abs(image), 65) # 65th percentile to eliminate any major spikes
    return np.nanstd(image[np.abs(image) < cont])


def select_from_position_axis(cube, racen: float, deccen: float, awidth: float, wcs: WCSClass):
    """Select out data from the cube.

    racen: float
        The decimal deg of the ra
    awidth: float
        The decimal deg of the positional width

    """
    awidth = int(awidth / 2)
    xcenl = wcs(racen + awidth, 'pix', 'ra---sin')
    xcenr = wcs(racen - awidth, 'pix', 'ra---sin')
    ycenl = wcs(decen - awidth, 'pix', 'dec--sin')
    ycenr = wcs(decen + awidth, 'pix', 'dec--sin')
    cut = cube[xcenl:xcenr, ycenl:ycenr, ...]
    return cut


def select_from_spectral_axis(cube, velcen: float, vwidth: float, wcs: WCSClass):
    """Select out data from the cube.

    racen: float
        The decimal deg of the ra
    awidth: float
        The decimal deg of the positional width

    """
    vwidth = int(vwidth / 2)
    vcenl = wcs(velcen - vwidth, 'pix', 'freq')
    vcenr = wcs(velcen + vwidth, 'pix', 'freq')
    cut = cube[:, :, vcenl:vcenr]
    return cut


def sum_image(image, width: int = None, axis: int = 0):
    if axis == 2:
        image = image.T
    if width is not None:
        width = int(width / 2)
        center = list(map(lambda x: int(x / 2), image.shape))
        summed_image = np.sum(image[center[0] - width:center[0] + width, ...], axis=0)  # noqa
    else:
        summed_image = np.sum(image, axis=0)
    return summed_image


# find a better spot for WCS. Probably in io, or need a coordinates library
# also there is misc trash in nkrpy.io.files.functions regarding header generation

def k_2_jy(freq: float, theta_major: float,
           theta_minor: float, brightness: float) -> float:
    """Convert Kelvin to Jy.

    Parameters
    ----------
    Parameters
    ----------
    freq: float
        ghz
    theta_major: float
        arcseconds
    theta_minor: float
        arcseconds
    brightness: float
        Kelvin/beam.
    """
    conv = (1.222E3 * (freq ** -2) / theta_minor / theta_major) ** -1
    return brightness * conv


def jy_2_k(freq: float, theta_major: float,
           theta_minor: float, intensity: float) -> float:
    """Convert Kelvin to Jy.

    Parameters
    ----------
    freq: float
        ghz
    theta_major: float
        arcseconds
    theta_minor: float
        arcseconds
    intensity: float
        mJy/beam

    """
    conv = 1.222E3 * (freq ** -2) / theta_minor / theta_major
    return intensity * conv


@validate
def convert_file(filename: str, jy_k: bool = False, k_jy: bool = False) -> tuple:
    """Convert a file between the types jy and kelvin.

    File must have restfrq bmaj bmin bunit defined

    Parameters
    ----------
    filename: str
        name of the file to convert. Must
    jy_k: bool
        default False, convert from jy to kelvin
    k_jy: bool
        default False, convert from kelvin to jy

    Returns
    -------
    tuple
        "conversionType_oldFilename", newHeader, new3DData

    """
    assert not (jy_k and k_jy)
    header, data = nkrpy_fits.read(filename)
    from nkrpy.astro import WCS
    wcs = WCS(header)
    data = np.squeeze(data)
    print(wcs.get_beam())
    inp = [float(wcs.get_head('restfreq')) / 1E9,
           wcs.get_beam()[0]*3600,
           wcs.get_beam()[1]*3600]
    print(f'Flux density unit: {wcs.get_head("bunit")}')
    print(f'RFREQ: {inp[0]}GHZ\nBeam: {inp[1]}x{inp[2]}')
    unit = str(wcs.get_head('bunit')).split('/')[0]
    print(wcs.header)
    if unit == 'Jy' or unit == 'K':
        cv = 1000
    elif unit == 'mJy' or unit == 'mK':
        cv = 1

    if jy_k:
        nd = jy_2_k(*inp, data * cv)
        nh = 'K/beam'
    elif k_jy:
        nd = k_2_jy(*inp, data) / cv
        nh = 'Jy/beam'
    if nh:
        fname = f'{nh.replace("/", "_")}_{filename}'
        header['BUNIT'] = nh
        nkrpy_fits.write(fname, header=header, data=nd)
        return fname, header, nd
    return None, None, None




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


def blackbody_hz(nu_hz, temperature_k):
    intensity = 2. * _h * nu_hz ** 3 / _c ** 2
    intensity *= 1. / (np.exp(_h * nu_hz / (_kb * temperature_k)) - 1)
    return intensity


def blackbody_cm(lam_cm, temperature_k):
    intensity = 2. * _h * _c ** 2 / lam_cm ** 5
    intensity *= 1. / (np.exp(_h * _c / (_kb * lam_cm * temperature_k)) - 1)
    return intensity


def opticaldepth_attenuation(tau):
    return np.exp(-tau)


def opacity_attenuation(kappa_cm_g, columndensity_cm, mass_h2_amu=2.8, dgr=0.01):
    tau = mass_h2_amu * _mh * columndensity_cm / dgr
    return opticaldepth_attenuation(tau)


def integrate_sed(vu=[0, np.inf], function=blackbody_hz, **kwargs):
    def intfunc(nu):
        return function(nu, **kwargs)
    result = quad(intfunc,sorted(vu), full_output=True)
    return result


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



if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
