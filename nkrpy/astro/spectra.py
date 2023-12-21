"""."""
# flake8: noqa
# cython modules

# internal modules
import warnings

# external modules
import numpy as np
import scipy
from scipy.signal import savgol_filter
from scipy.interpolate import InterpolatedUnivariateSpline

# relative modules
from ._wcs import WCS
from ._functions import (select_ellipse, select_rectangle, select_conic_section, select_circle, select_circular_annulus, select_elliptical_annulus)
from .. import math as nkrpy_math
from ..io import Log as Logger
from .._types import LoggerClass

# global attributes
__all__ = ['extract_components_from_spectra', 'extract_spectra_from_datacube']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
warnings.simplefilter('ignore')


def extract_components_from_spectra(d, sampler: int = None, logger: LoggerClass = None, minpolyfit: int = 1, maxpolyfit: int = 15):
    """Spectral Component Extractor

    This makes several assumptions but tries to fit and extract out the continuum vs line components and give a local rms estimator.

    It doesn't attempt to scientifically fit the continuum, rather just extract this component.

    Assumptions
    -----------
    > data is well sampled (R > 3000)
    > data is continuum dominated
    > noise is gaussian or at least symmetric and random
    > data is relatively well ordered and semi-regularly spaced (breaks are allowed but fits become less stable)

    Parameters
    ----------
    d: np.ndarray
        data[orders, (lam, flux, ...), numpoints]
    sampler: int
        number of units for sampling, should be of order 5 * units across line emission. This is a hyper parameter that constrains the subsequent smoothing and binning. Default 5 * median resolution / smallest resolution. In the case of uniform spectral sampling, sampling is 5 spectral units. Smaller number is more aggressive extraction and takes longer.

    Returns
    -------
    lam, cont, line, rms: np.ndarray
        The spectralunit, continuum component, line emission component, and computed local rms error with the same demisions as the input data.
    [fitcont]
        Returns the fit continuum with a poly fit. numpy.poly1d

    Usage
    -----
    # d  is data of shape (order, (lam, flux, ...), data)
    lam, cont, line, rms = extract_components(d)                         
    # Sampler:  ####

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots() 
    ax.set_xlim(1.15, 2.45) 
    ax.set_ylim(0, 1.4e-15) 
    ax.plot(lam, cont+line, color='black',zorder=1) 
    ax.plot(lam, cont, color='blue',zorder=20) 
    ax.errorbar(x=lam, y=cont, yerr=rms, color='yellow', alpha=0.1)
    plt.show()

    """
    if logger is None:
        logger = Logger
    logger.setup()
    ndata = np.concatenate(d, axis=1)
    ndata = ndata[..., ndata[0, :].argsort()]
    d = d.T
    mask = ndata[0, :] > 0
    mask *= ~np.isnan(ndata[0, :])
    data = ndata[:, mask]
    lowercut = np.ravel(np.where(~np.isnan(data[1, :])))[0]
    uppercut = np.ravel(np.where(~np.isnan(data[1, :])))[-1]
    data = data[:, lowercut:uppercut + 1]
    # make high resolution
    # resample to same resolution
    logger.debug(f'''
        Data: 
            shape: {data.shape}
            lower: {np.min(data[0, :])}
            upper: {np.max(data[0, :])}
            ''')
    fineres = np.full([4, data.shape[-1]], np.nan, dtype=float)
    fineres[:3, :] = data[:]
    fineres = fineres[:, fineres[0, :].argsort()]
    reorder = np.searchsorted(fineres[0, :], np.unique(fineres[0, :]))
    fineres = fineres[:, reorder]
    fineres[-1, :] = fineres[1, :]
    # first SG smooth, pull put high freq noise
    window = fineres.shape[-1] // sampler
    window += ((window + 1) % 2)
    logger.debug(f'''First SG Smoothing, window: {window}''')
    fineres[-1, :] = savgol_filter(fineres, axis=1, window_length=window, polyorder=2)[1, :]
    #from IPython import embed; embed()
    # Initial Non aggressive Filter data using 2 sigma
    normed = np.abs(fineres[-1, :] - fineres[1, :])
    mk = np.isnan(normed)
    mx = np.percentile(normed[~mk], [95])
    mk += normed > mx
    spline = InterpolatedUnivariateSpline(x=fineres[0, ~mk], y=fineres[-1, ~mk], k=1)
    fineres[-1, :] = spline(fineres[0, :])
    fineres[-1, :] = savgol_filter(fineres, axis=1, window_length=window * 2 + 1, polyorder=2)[-1, :]
    # Aggressive filter using 1 sigma
    normed = np.abs(fineres[-1, :] - fineres[1, :])
    mk = np.isnan(normed)
    mx = np.percentile(normed[~mk], [65])
    mk += normed > mx
    fineres[-1, mk] = np.nan

    # spline fit again to interpolate between continuum and desample resolution
    spline = InterpolatedUnivariateSpline(x=fineres[0, ~mk], y=fineres[-1, ~mk], k=1)
    fineres[-1, :] = spline(fineres[0, :])

    wave = fineres[0, :]
    continuum_flux = fineres[-1, :]
    line_flux = fineres[1, :]
    normed = np.abs(line_flux - continuum_flux)
    weights = fineres[2, :]
    nans = np.isnan(continuum_flux)
    nans += np.isnan(line_flux)
    nans += np.isnan(weights)
    mx = np.percentile(normed[~nans], 65)
    fitregion = normed[~nans] < mx
    ps = []
    rmslast = -np.inf
    for i in range(max([minpolyfit, 0]), maxpolyfit + 1):
        coef = np.polyfit(wave[~nans][fitregion], continuum_flux[~nans][fitregion], w= 1. / weights[~nans][fitregion], deg=i)
        polyfunc = np.poly1d(coef)
        fitcont = polyfunc(wave)
        res = np.nanstd(fineres[1, :][~nans][fitregion] - fitcont[~nans][fitregion])
        ps.append([res * (i + 1), polyfunc])
        logger.debug(f"""Fit {i} order poly:
            res={res:0.2e}
            params={coef}""")
    if len(ps) > 0:
        ps.sort(key=lambda x: x[0])
        bestpoly = ps[0]

        logger.debug(f'Continuum fit weighted error: {bestpoly[0]: 0.2e}')
        rms, bestpoly = bestpoly
    else:
        bestpoly = None
    return {'x': wave, 'continuum': continuum_flux, 'line': line_flux, 'weight': 1. / weights, 'polyfit': bestpoly}


def extract_spectra_from_datacube(cube: np.ndarray, racen: float, deccen: float, pa: float, smaj: float, smin: float, wcs: WCS, dtype: str = 'circular_annulus'):
    """
    cube goes pos, pos,freq
    racen, deccen are in same units as axis
    axis in degrees
    PA goes E of north (counterclockwise)
    ."""
    cube = cube.copy()
    pos = -pa * np.pi / 180.
    data = np.squeeze(cube)
    racenpix = wcs(racen, 'pix', wcs.axis1['type'])
    deccenpix = wcs(deccen, 'pix', wcs.axis2['type'])
    delt = wcs.axis1['delt']
    smajpix, sminpix = np.abs(smaj/(delt * 3600)), np.abs(smin/ (delt*3600))
    mask = select_ellipse(data.shape[:-1], xcen=racenpix, ycen=deccenpix, sma=smajpix, smi=sminpix, pa=pos)
    internal = np.nansum(data[mask], axis=0)
    external = np.nansum(data[~mask], axis=0)
    return internal, external


def test():
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
