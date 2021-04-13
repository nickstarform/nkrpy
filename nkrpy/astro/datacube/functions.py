"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import numpy as np

# relative modules
from ..._types import WCSClass
from ...misc.errors import ArgumentError

# global attributes
__all__ = ('select_from_position_axis', 'cube_rms')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def cube_rms(cube: np.ndarray, N: int = 5):
    """Calulcate the rms of a cube based on the edge channels."""
    x1, x2 = np.percentile(np.arange(cube.shape[1]), [25, 75])
    y1, y2 = np.percentile(np.arange(cube.shape[0]), [25, 75])
    x1, x2, y1, y2 = map(int, [x1, x2, y1, y2])
    if cube.shape[-1] < 2 * N:
        N = int(np.floor(cube.shape[-1] / 2))
    rms = np.nanstd([cube[y1:y2, x1:x2, :N], cube[y1:y2, x1:x2, -N:]])
    return rms


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



def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code

# end of file
