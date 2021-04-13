"""."""
# cython modules

# internal modules

# external modules
import numpy as np

# relative modules
from ..misc import freq_vel, vel_freq
from ... import constants
from ... import WCSClass
from .._wcs import WCS
from ... import Unit
from .functions import select_from_spectral_axis
from .functions import cube_rms

# global attributes
__all__ = ('momentmap', 'momentmap_km', 'test')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
MIN_MOMENT = -3

# work on this
def momentmap_km(cube: np.ndarray, wcs: WCSClass, velstart: float, velstop: float, restfreq: float = None, moment: int = 0, v_sys: float = 0, find_nearest: bool = False):
    """Make a moment map based on velocity ranges.

    velstart in km

    Data should be pos, pos, spectral (freq or wavelength)
    restfreq in hz
    vel = -1. * (freq - rf) / rf * constants.c / 100. / 1000.
    freq = restfreq - vel / c * restfreq
    """
    # convert from WCS into PIX
    ret = None, None, None
    cube = np.squeeze(cube)
    restfreq = restfreq if restfreq is not None else wcs.get_head('restfreq')
    #from IPython import embed; embed()
    
    startf = vel_freq((velstart + v_sys) * 1000. * 100., restfreq)
    stopf = vel_freq((velstop + v_sys) * 1000. * 100., restfreq)
    startf, stopf = (startf, stopf) if startf < stopf else (stopf, startf)
    start = wcs(startf, 'pix', 'freq', find_nearest=False)
    stop = wcs(stopf, 'pix', 'freq', find_nearest=False)
    if start < 0 and not find_nearest:
        start = 0
    elif start < 0:
        return ret
    if stop < 0 and not find_nearest:
        stop = 0
    elif stop < 0:
        return ret
    if stop < start:
        start, stop = (start, stop) if start < stop else (stop, start)
    if stop > cube.shape[-1]:
        stop = cube.shape[-1] - 2
    cen = wcs(vel_freq(v_sys * 1000. * 100., restfreq), 'pix', 'freq')
    rfcen = wcs(restfreq, 'pix', 'freq')
    if (stop - start) == 0:
        return ret
    start, stop = map(int, [start, stop])
    #from IPython import embed; embed()
    spectralaxis = np.arange(start, stop + 1).astype(int)
    spectralaxis = wcs(spectralaxis, 'wcs', 'freq')
    spectralaxis = freq_vel(spectralaxis, restfreq) / 1000. / 100.
    if (stop + 1 - start) != spectralaxis.shape[0]:
        return ret
    #print(f'RFcen {rfcen:0.1f}, vsys {v_sys:0.1f}kms@{cen:0.1f}, start {velstart:0.1f}kms@{start}, stop {velstop:0.1f}kms@{stop}')
    mx, dmx = momentmap(cube=cube[..., start:stop + 1], vel_axis=spectralaxis, moment=moment, wcs=wcs)
    newwcs = WCS(wcs)
    newwcs.get('freq')['rval'] = (startf + stopf) / 2.
    newwcs.get('freq')['rpix'] = 1
    newwcs.get('freq')['axis'] = 1
    newwcs.update_header()
    vel = freq_vel(newwcs.get('freq')['rval'], restfreq) / 1000. / 100.
    return mx, dmx, newwcs


def momentmap(cube: np.ndarray, freq_axis: np.ndarray = None, vel_axis: np.ndarray = None, wcs: WCSClass = None, moment: int = 0, restfreq: float = None):
    """General wrapper for creating moments of a datacube.

    Parameters
    ----------
    cube: np.ndarray
        The 3d datacube. A nparray of pos, pos, freq
    freq_axis: np.ndarray
        The 1d array describing the freq axis. required for moment >= 1
    moment: int
        This has to be >= {MIN_MOMENT}

    Returns
    -------
    np.ndarray
        The image of the datacube using the selected moment map.

    Usage
    -----

    Moments
    -------
        -3: the stdev along the freq axis
        -2: the average along the freq axis
        -1: the median along the freq axis
        0: integrate along the freq axis
        1: velocity weighted
        2: velocity dispersion
        3+: ...
    """
    cube = np.squeeze(cube)

    def _minus3(cube: np.ndarray):
        return np.nanstd(cube, axis=2), None

    def _minus2(cube: np.ndarray):
        return np.nanmean(cube, axis=2), None

    def _minus1(cube: np.ndarray):
        return np.nanmedian(cube, axis=2), None

    def _zeroth(cube: np.ndarray, vel_axis: np.ndarray):
        chan = np.diff(vel_axis).mean()
        npix = np.nansum(cube != 0.0, axis=2)
        m0 = np.trapz(cube, x=vel_axis, axis=2)
        rms = cube_rms(cube)
        disp = chan * rms * npix ** 0.5 * np.ones(m0.shape)
        return m0, disp

    def _first(cube: np.ndarray, vel_axis: np.ndarray):
        vel_axis = vel_axis[None, None, :] * np.ones(cube.shape)
        weights = 1e-10 * np.random.rand(cube.size).reshape(cube.shape)
        weights = np.where(cube != 0.0, np.abs(cube), weights)
        m1 = np.average(vel_axis, weights=weights, axis=2)
        npix = np.sum(cube !=0., axis=2)
        m1 = np.where(npix >= 1., m1, np.nan)
        rms = np.nanstd(m1)
        #print(f'RMS:{rms:0.1e}')
        return m1, None

    def _second(cube: np.ndarray, vel_axis: np.ndarray):
        vpix = vel_axis[None, None, :] * np.ones(cube.shape)

        weights = 1e-10 * np.random.rand(cube.size).reshape(cube.shape)
        weights = np.where(cube != 0.0, abs(cube), weights)
        rms = cube_rms(cube)

        #from IPython import embed; embed()
        m1, _ = _first(cube=cube, vel_axis=vel_axis)
        m1 = m1[:, :, None] * np.ones(cube.shape)
        m2 = np.sum(weights * (vpix - m1)**2, axis=2) / np.sum(weights, axis=2)
        m2 = np.sqrt(m2)
        #dm2 = ((vpix - m1)**2 - m2**2) * rms / np.sum(weights, axis=2)
        #dm2 = np.sqrt(np.sum(dm2**2, axis=2)) / 2. / m2

        npix = np.sum(cube != 0.0, axis=2)
        m2 = np.where(npix >= 1.0, m2, np.nan)
        return m2, None #dm2

    def _nth(cube: np.ndarray, freq_axis: np.ndarray, n: int):
        m0, _ = _zeroth(cube, freq_axis)
        return np.trapz((cube * freq_axis) ** n, x=freq_axis, axis=2) / m0, None

    assert moment >= MIN_MOMENT
    rf = restfreq
    if wcs is not None:
        rf = rf if rf is not None else wcs.get_head('restfreq')
        if vel_axis is None and freq_axis is None:
            freq_axis = wcs.array(return_type='wcs', axis='freq')
            vel_axis = freq_vel(freq_axis, rf) / 1000. / 100.
        elif freq_axis is not None:
            vel_axis = freq_vel(freq_axis, rf) / 1000. / 100.
    else:
        if vel_axis is None:
            print('If freq_axis is specified, then restfreq must be provided.')
            sys.exit()


    if np.diff(vel_axis).mean() < 0:
        cube = cube[::-1]
        vel_axis = vel_axis[::-1]
    vel_axis *= 1000.
    cuberms = cube_rms(cube)
    #print(f'Cube RMS: {cuberms:0.2e}, {np.diff(vel_axis).mean()}, {vel_axis[0]}, {cube.shape}, {vel_axis.shape}')
    #print('Int from ', vel_axis[0], ' to ', vel_axis[-1])
    if moment > 0 and vel_axis is not None:
        assert cube.shape[-1] == vel_axis.shape[-1]
    if moment == -3:
        mom, mx = _minus3(cube)
    elif moment == -2:
        mom, mx = _minus2(cube)
    elif moment == -1:
        mom, mx = _minus1(cube)
    elif moment == 0:
        mom, mx = _zeroth(cube, vel_axis)
    elif moment == 1:
        mom, mx = _first(cube, vel_axis)
    elif moment == 2:
        mom, mx = _second(cube, vel_axis)
    elif moment > 2:
        mom, mx = _nth(cube, vel_axis, moment)
    return mom, mx

momentmap.__doc__ = momentmap.__doc__.replace('MIN_MOMENT', f'{MIN_MOMENT}')

def test_backend_setup():
    a = np.random.random((3, 3, 4))
    b = np.array([100, 105, 110, 115], dtype=np.float64)
    return a, b


def test_backend_repeat(cube: np.ndarray, freq_array: np.ndarray, moment: int):
    _ = momentmap(cube, freq_array, moment)
    return None


def test():
    """Testing module."""
    import timeit
    number = 1000
    a = np.random.random((3, 3, 4))
    b = np.array([100, 105, 110, 115], dtype=np.float64)
    ashape = a.shape
    for i in range(MIN_MOMENT, 4, 1):
        t, _ = momentmap(a, b, i)
        if a.shape[:-1] != t.shape:
            print(f'Failed {moment}, shapes {cube.shape} vs {t.shape}')
            return False
        res = timeit.repeat(stmt=f"test_backend_repeat(a, b, {i})", setup="from __main__ import (test_backend_repeat, test_backend_setup); a, b = test_backend_setup()", number=number, repeat=10)
        res = np.min(res)
        print(f'Moment {i} took {res:0.1e}s')
    return True

if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    if test():
        print('Test Passed')
    else:
        print('Test Failed')

# end of code

# end of file
