"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
from scipy.
# relative modules
from .._wcs import WCS
from ...image.functions import (binning, select_ellipse)

# global attributes
__all__ = ('test', 'spectra')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def spectra(cube: np.ndarray, racen: float, deccen: float, pa: float, smaj: float, smin: float, wcs: WCS):
    """
    cube goes pos, pos,f freq
    racen, deccen are in same units as axis
    axis in degrees
    PA goes E of north (counterclockwise)
    ."""
    cube = cube.copy()
    pos = -1. * Unit(pa, 'degrees', 'radians').get_vals()
    data = np.squeeze(cube).T
    racenpix = wcs(racen, 'pix', 'ra---sin')
    deccenpix = wcs(deccen, 'pix', 'dec--sin')
    delt = wcs.get('ra---sin')['del']
    smajpix, sminpix = np.abs(smaj/(delt * 3600)), np.abs(smin/ (delt*3600))
    mask = select_ellipse(data.shape[:-1], racenpix, deccenpix, smajpix, sminpix, pos)
    data[~mask] = np.nan
    test = np.nansum(np.nansum(data, axis=0), axis=0)
    return test


def test():
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
