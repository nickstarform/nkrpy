"""."""
# flake8: noqa
# cython modules

# internal modules

# external modules
import numpy as np
from scipy.ndimage.interpolation import rotate
from IPython import embed

# relative modules

# global attributes
__all__ = ('binning', 'center_image', 'rotate_image', 'sum_image', 'select_ellipse')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def select_ellipse(datashape, xcen=None, ycen=None, sma=None, smi=None, pa=0):
    """
    pa in radians.
    """
    if xcen is None:
        xcen = datashape[0] / 2
    if ycen is None:
        ycen = datashape[1] / 2
    if sma is None:
        sma = datashape[0] / 2
    if smi is None:
        smi = sma
    # print(datashape, xcen, ycen, sma, smi, pa)
    x, y = np.indices(datashape)
    xp = (x - xcen) * np.cos(pa) + (y - ycen) * np.sin(pa)
    yp = -1. * (x - xcen) * np.sin(pa) + (y - ycen) * np.cos(pa)
    mask = ((xp / sma) ** 2 + (yp / smi) ** 2) <= 1
    return mask

def select_rectangle(data: np.ndarray, xcen: float = None, ycen: float = None, xlen: float = None, ylen: float = None, pa: float = 0, inplace: bool = False):
    """
    pa in radians.
    """
    datashape = data.shape[:2]
    if xcen is None:
        xcen = datashape[0] / 2
    if ycen is None:
        ycen = datashape[1] / 2
    if xlen is None:
        xlen = datashape[0] / 2
    if ylen is None:
        ylen = xlen
    # print(datashape, xcen, ycen, sma, smi, pa)
    x, y = np.indices(datashape)
    xp = np.abs(x - xcen) <= abs(xlen)
    yp = np.abs(y - ycen) <= abs(ylen)
    mask = xp * yp
    mask = rotate_image(mask.astype(float), angle=pa, resize=False, mode='constant', cval=np.nan, use_skimage=True)
    mask[mask<0.5] = np.nan
    mask[mask >=0.5] = 1
    if data.shape != mask.shape:
        mask = mask[..., None]
    if not inplace:
        return mask * data
    data *= mask


def remove_padding(data, pad_val = np.nan):
    if pad_val is np.nan:
        valid_rows = ~np.isnan(data).all(axis=1)
        valid_cols = ~np.isnan(data).all(axis=0)
    if len(valid_rows.shape) > 1:
        valid_rows = valid_rows[..., 0]
    if len(valid_cols.shape) > 1:
        valid_cols = valid_cols[..., 0]
    return data[valid_rows, ...][:, valid_cols, ...]


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


def center_image(image, ra, dec, wcs):
    xcen = wcs(ra, 'pix', 'ra---sin')
    ycen = wcs(dec, 'pix', 'dec--sin')
    imcen = list(map(lambda x: x / 2, image.shape[1:]))
    center_shift = list(map(int, [imcen[0] - ycen, imcen[1] - xcen]))
    shift = [0]
    shift.extend(center_shift)
    shifted_image = inter.shift(image, shift)
    return shifted_image


def rotate_image(image: np.ndarray, use_scipy: bool = False, use_skimage: bool = True, axis: int = 0, **kwargs):
    """

    Parameters
    ----------
    image: np.ndarray
    kwargs: dict
        Just passes all args into subsequent function
    """
    key = 'deg'
    for k in ['deg', 'angle', 'pa']:
        if k in kwargs:
            key = k
            break
    deg = kwargs[key]
    if axis != 0:
        image = image.T
    rotated_image = image.astype(np.float64)
    if deg % 90 == 0:
        for _ in range(deg / 90):
            rotated_image = np.rot90(rotated_image, **kwargs)
    else:
        if use_scipy:
            from scipy.ndimage import interpolation as inter
            # counter clockwise
            func = inter.rotate
        elif use_skimage:
            from skimage import transform
            # counter clockwise
            func = transform.rotate
        rotated_image = func(rotated_image, **kwargs)
    if axis == -1:
        rotated_image = rotated_image.T
    return rotated_image


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
