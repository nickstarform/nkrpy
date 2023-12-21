"""Various math functions."""
# cython: binding=True
# cython modules
cimport numpy as cnp  # noqa
cimport cython  # noqa

# standard modules
from itertools import chain, islice, groupby
from operator import itemgetter
from copy import deepcopy

# external modules
import numpy as np
from scipy import special
from decimal import Decimal
from mpmath import mp

# relative modules
from ..misc import constants as n_c
from ..misc import functions as n_f

pi = n_c.pi
typecheck = n_f.typecheck
# global attributes
__all__ = ['flatten', 'listinvert', 'binning', 'cross', 'tolerance_split', 'normalize', 'windowrms',
           'dot', 'radians', 'deg', 'mag',
           'ang_vec', 'determinant', 'inner_angle',
           'angle_clockwise', 'rolling_average_value', 'rolling_average_points', 'list_array',
           'pairwise', 'window', 'group_ranges', 'rms', 'sigma', 'rotate_rectangle', 'rotate_point', 'ellipse_distance', 'width_convolution_kernel', 'gaussian_smoothing_kernel', 'chi2']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def chi2(observed: np.ndarray, model: np.ndarray, number_of_fit_params: int=None):
    x2 = np.nansum((observed - model) / np.var(observed))
    return x2 if number_of_fit_params is None else x2 / (observed.shape[0] - number_of_fit_params)


def width_convolution_kernel(x, xcen, a: float = -2, b: float = 10):
    '''Gaussian Convolution Kernel Width'''
    return a * np.abs(x - xcen) + b


def gaussian_smoothing_kernel(kernelshape: list, xc: float = None, yc: float = None, sigma_x: float = 1, sigma_y: float = 1, degrees: float = 0):
    #This function returns a Gaussian smoothing kernel the size of the 2D input
    #array. The center of the Gaussian is at (xc, yc), the sigma's in the x and
    #y direction are given by sigma_x and sigma_y (all in pixels), the rotation 
    #of the Gaussian is given by rotation in degrees.
      
    #Dimensions of the input array.        
    nx, ny = kernelshape
    nx = nx if nx % 2 == 1 else nx + 1
    yx = ny if ny % 2 == 1 else ny + 1
    if xc is None:
        xc = nx // 2 + 1
    if yc is None:
        yc = ny // 2 + 1
    #Creating an coordinate grid.
    y, x = np.mgrid[0:nx, 0:ny]
    #Change rotation angle to radians.
    rot = degrees * np.pi / 180.

    #Lets first rotate the coordinates.
    xr = x * np.cos(rot) - y * np.sin(rot) 
    yr = x * np.sin(rot) + y * np.cos(rot)
    #Rotate the center
    xcr = xc * np.cos(rot) - yc * np.sin(rot) 
    ycr = xc * np.sin(rot) + yc * np.cos(rot)
    
    #Creating the 2D Gaussian.
    kernel = np.exp(-(((xr - xcr) / sigma_x) ** 2 + ((yr - ycr) / sigma_y) ** 2) / 2)
    return kernel / np.sum(kernel)


def rotate_point(x, y, angle):
    # calulate the bounding box for a rectangle rotated clockwise about center
    angle = angle % 360
    radians = angle * np.pi / 180.
    pi2 = np.pi / 2.
    X = x * np.cos(radians) + y * np.sin(radians)
    Y = -x * np.sin(radians) + y * np.cos(radians)
    return X, Y


def rotate_rectangle(w, h, angle):
    # calulate the bounding box for a rectangle rotated clockwise about center
    angle = angle % 360
    radians = angle * np.pi / 180.
    H = abs(w * np.sin(radians)) + abs(h * np.cos(radians))
    W = abs(h * np.sin(radians)) + abs(w * np.cos(radians))
    return W, H


def sigma(single: int = None, start: int = None, stop: int = None, interval: int = None):
    if single is not None:
        start = single
        stop = single
        interval = 1
    sigmas = []
    for sig in np.arange(start, stop + interval, interval):
        sigmas.append(mp.erf(sig / mp.sqrt(2)))
    return np.ravel(sigmas)


def windowrms(x, width: int = 11):
    """Compute the RMS of array within window.

    width: int
        Must be odd. If even will force to odd + 1
    """
    width += (width + 1) % 2
    bg = np.zeros(x.shape, dtype=float)
    for iv, vv in enumerate(x):
        left = 0 if iv < width // 2 else iv - width // 2
        right = len(x) - 1 if iv > (len(x) - 1 - width // 2) else (iv + width // 2)
        bg[iv] = np.nanstd(x[left:right])
    return bg


def normalize(x, norm_median: float = 1, norm_max: float = None, norm_min: float = None):
    """Normalize array.

    Parameters
    ----------
    norm_median: float
        If given will normalize array so that the median is this value
    norm_max: float
        if given then norm_min must be given. Will normalize array to fall between these values.

    Returns
    -------
    np.ndarray of len(x)
    """
    med = np.nanmedian(x)
    if norm_median is not None:
        return x / med + (norm_median - 1.)
    norm_max, norm_min = (norm_max, norm_min) if norm_max > norm_min else (norm_min, norm_max)
    return (x / np.nanmax(x)) * (norm_max - norm_min) + norm_min


def rms(observed: np.ndarray, predicted: np.ndarray = None, axis: int = 0, dof: int = 0):
    predicted = predicted if predicted is not None else np.zeros(observed.shape)
    num = observed.shape[axis]
    return np.nansum((observed - predicted) ** 2) ** 0.5 / (num - dof)


def ellipse_distance(major: float,
                     minor: float,
                     ellipse_pa: float = 0,
                     vector_pa: float = 0,
                     x_offset: float = 0,
                     y_offset: float = 0,
                     vector_ref: bool = False):
    """Calculate distance from center to edge of ellipse at an angle.

    Angles defined east of north.
    Parameters
    ----------
    Major, minor: float
        The major and minor axis of the ellipse
    x_offset, y_offset: float
        The offset of the ellipse from the origin
    ellipse_pa: float
        The position angle in degrees the ellipse is rotated. This is defined East of North where the major axis is oriented North
    vector_pa: float
        The angle of the vector east of north wrt the ellipse
    vector_ref: bool
        Whether the vector references the ellipse or the origin (default)
    Returns
    -------
    distance: tuple
        Distance to the point along the ellipse
    """
    # reduce angles to 0 -> 360
    if not vector_ref:
        vector_pa = ellipse_pa - vector_pa
    ellipse_pa += 90.
    vector_pa = vector_pa % 360.
    ellipse_pa = ellipse_pa % 360.
    # turn to radians
    radians = lambda deg: deg * np.pi / 180.  # noqa
    theta, phi = map(radians, [ellipse_pa, vector_pa])
    cosp = np.cos(phi)
    sinp = np.sin(phi)
    cost = np.cos(theta)
    sint = np.sin(theta)
    y = major * cosp * cost - minor * sinp * sint + y_offset
    x = major * cosp * sint + minor * sinp * cost + x_offset
    distance = lambda x1, x2, y1, y2: np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)  # noqa
    return distance(x, 0, y, 0), distance(x, x_offset, y, y_offset)


def group_ranges(data):
    """Yield range of consecutive numbers."""
    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        group = (map(itemgetter(1), g))
        group = list(map(int, group))
        if len(group) == 0:
            yield (group[0], )
        yield (group[0], group[-1])


def tolerance_split(input_list, tolerance, sorted: bool=True):
    """Yield successive values in list that are consecutive within tolerance.

    If not sorted, then a copy of the input list is made and sorted.
    If input_list is a 2d list, then will compare first elements only."""
    if len(input_list) == 0:
        return
    if not sorted:
        input_list = deepcopy(input_list)
        input_list.sort()
    res = []
    last = input_list[0]
    for ele in input_list:
        if typecheck(ele) and len(ele) > 1:
            if ele[0] - last[0] > tolerance:
                yield res
                res = []
            res.append(ele)
            last = ele
        else:
            if ele - last > tolerance:
                yield res
                res = []
            res.append(ele)
            last = ele
    yield res


def pairwise(points, window_points: int = 2):
    """Create point pairs given a window.

    So [1,2,3,4, 5] with window 2 (e.g. 2 points) yields
        [[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]]
    whereas with window 4 yields
        [[1.0, 4.0], [2.0, 5.0]]
    """
    window_points -= 1
    if isinstance(points, list):
        for x in window(points, window_points):
            yield x[0], x[-1]

    if isinstance(points, np.ndarray):
        for x in __window_array(points, window_points):
            yield x[0], x[-1]


def window(points, window_points: int = 2):
    if isinstance(points, list):
        return __window_list(points, window_points)

    if isinstance(points, np.ndarray):
        return __window_array(points, window_points)


def __window_array(points: np.ndarray, window_points: int = 2):
        ind = np.arange(0, points.shape[0] + 1).tolist()
        ind = np.ravel(list(zip(ind, ind[window_points:])))
        spl = np.split(points, ind)
        if spl[0].shape[0] == 0:
            spl = spl[1::2]
        else:
            spl = spl[::2]
        for elem in spl:
            yield elem


def __window_list(points: list, window_points: int = 2):
    """Similar to pairwise, but efficiently returns window."""
    it = iter(points)
    result = tuple(islice(it, window_points))
    if len(result) == window_points:
        yield result
    for elem in it:
        result = result[1:] + (elem, )
        yield result


def flatten(lol, ret: list = [], inplace: bool = True):
    if not inplace:
        return np.ravel(lol)
    if len(lol) == 0:
        ret.append(lol)
        return
    for l in lol:
        if typecheck(l):
            flatten(l, ret)
        else:
            ret.append(l)
    return ret


def binning(data, width=3):
    """Bin the given data along the 0th axis.
    """
    if width == 1:
        return data[:]
    if len(data.shape) == 1:
        data = data[..., None]
    binned = np.zeros((data.shape[0] // width, data.shape[-1])).astype(data.dtype)
    for axis in range(data.shape[-1]):
        tobin = data.ravel()[slice(axis, None, data.shape[-1])]
        binned[:, axis] = np.nanmean(tobin[:(tobin.size // width) * width].reshape(-1, width), axis=1)
    return np.squeeze(binned)


def listinvert(total, msk_array):
    """Invert list with mask.

    msk_array must be the index values
    """
    mask_tot = np.full(total.shape, True, dtype=bool)
    mask_tot[msk_array] = False
    return mask_tot


def cross(a, b):
    """Compute cross product between a, b."""
    assert len(a) == len(b)
    if len(a) == 3:
        c = (a[1] * b[2] - a[2] * b[1],
             a[2] * b[0] - a[0] * b[2],
             a[0] * b[1] - a[1] * b[0])
    elif len(a) == 2:
        c = (0, 0, a[0] * b[1] - a[1] * b[0])
    return c


def dot(a, b):
    """Compute dot product between a and b."""
    ret = 0
    for i in range(len(a)):
        ret += a[i] * b[i]
    return ret


def radians(d):
    """Convert degrees to radians."""
    return d / 180. * pi


def deg(r):
    """Convert radians to degrees."""
    return r * 180. / pi


def mag(a):
    """Mag of a vector."""
    ret = 0
    for x in a:
        ret += x ** 2
    return (ret) ** 0.5


def ang_vec(deg):
    """Generate a unit vec of deg."""
    rad = deg * pi / 180.
    return np.cos(rad), np.sin(rad)


def determinant(v, w):
    """Determine the determinant of two vecs."""
    return v[0] * w[1] - v[1] * w[0]


def inner_angle(v, w):
    """Calculate inner angle between two vecs."""
    cosx = dot(v, w) / (mag(v) * mag(w))
    while np.abs(cosx) >= 1:
        cosx = cosx / (np.abs(cosx) * 1.001)
    rad = np.arccos(cosx)  # in radians
    return rad * 180. / pi  # returns degrees


def angle_clockwise(a, b):
    """Determine the angle clockwise between vecs."""
    inner = inner_angle(a, b)
    det = determinant(a, b)
    if det > 0:  # this is a property of the det.
        # If the det < 0 then B is clockwise of A
        return inner
    else:  # if the det > 0 then A is immediately clockwise of B
        return 360. - inner


def rolling_average_value(ilist: np.ndarray, window: float):
    """Apply a given constant window to a list.

    Parameters
    ----------
    ilist: np.ndarray
        list of values to apply window to
    window: float
        value in the same units as the list. Will return
        a list where any values within a window will be
        averaged together and returned

    Return
    ------
    list
        windowed list

    """
    if len(ilist.shape) == 1:
        original = ilist[..., None]
    else:
        original = ilist[:]
    original = original[np.argsort(original[..., 0])]
    rolling_avg = []
    ci = 0
    cval = original[0, 0]
    while ci < original.shape[0]:
        cmask = original[:, 0] < cval
        cmask += original[:, 0] > cval + window
        rolling_avg.append(np.nanmean(original[~cmask, ...], axis=0))
        l = np.ravel(np.where(~cmask))
        if np.max(l) == ci:
            ci += 1
        else:
            ci = np.max(l)
        if ci < cmask.shape[0]:
            cval = original[ci, 0]
    if np.max(l) < cmask.shape[0] - 1:
        rolling_avg.append(np.nanmean(original[np.max(l):, ...], axis=0))
    return np.squeeze(np.array(rolling_avg))


def rolling_average_points(ilist: np.ndarray, window_width: int, fill_value: float = None, end: str = 'stretch'):
    """Apply a given constant window to a list. Only 1d and 2d array supported. Sorts by zeroth axis

    Parameters
    ----------
    ilist: np.ndarray
        list of values to apply window to
    window_width: float
        value in the same units as the list. Will return
        a list where any values within a window will be
        averaged together and returned
    fill_value: float
        The value to fill in when summing. Only useful if end = 'pad'
    end: str
        Allowed values: stretch cyclic pad bounded
         * bounded: forces the sums to be made from strictly increasing values from 0-array.shape
         * stretch: similar to bounded, but preserves original array shape by decreasing window_width at edges
         * cyclic: preserves window_width and array shape by cycling the array at the endpoints
         * pad: preserves window_width and array shape by padding array at the endpoints by the defined value. Setting fill_value to None or nan is equivalent to stretch

    Return
    ------
    meaned np.ndarray
        windowed list

    """
    assert end in ['stretch', 'cyclic', 'pad', 'bounded']
    if len(ilist.shape) == 1:
        original = ilist[..., None]
    else:
        original = ilist[:]
    original = original[np.argsort(original[..., 0])]
    if end == 'stretch':
        # if stretch creates an array of the same shape. The ends are less sensitive than the center
        indices = np.arange(original.shape[0], dtype=int)
        indices = list(window(indices, window_width))
        for i in list(range(1, window_width - 1)):
            if i > window_width // 2:
                continue
            indices = [np.arange(0, window_width - i, dtype=int)] + indices
        for i in list(range(2, window_width))[::-1]:
            if i <= window_width // 2:
                continue
            indices += [np.arange(original.shape[0] - i, original.shape[0], dtype=int)]
        return np.squeeze(np.array([np.nanmean(original[i, ...], axis=0) for i in indices]))
    elif end in ['cyclic', 'pad']:
        # if cyclic will cycle back through the points to the other side of the array
        indices = np.linspace(-window_width // 2 + 1, original.shape[0] + window_width// 2, num=original.shape[0] + window_width - 1, dtype=int)
        indices = window(indices, window_width)
    elif end == 'bounded':
        # if bounded creates an array of the shape input.shape - window_width - 1, where the sumed values must be strictly increasing within the array
        rolling_avg = np.nanmean(list(window(original, window_width)), axis=1)
        return np.squeeze(rolling_avg)
    rolling_avg = []
    summed = np.full([window_width, *list(original.shape[1:])], fill_value, dtype=float)
    for i in indices:
        summed[:] = fill_value
        if end == 'pad' and any(i < 0):
            mask = i < 0
            summed[~mask] = original[i[~mask] % 15, ...]
        else:
            summed[:] = original[i % 15, ...]
        rolling_avg.append(np.nanmean(summed, axis=0))
    return np.squeeze(np.array(rolling_avg))


def _1d(ite, dtype):
    """Create 1d array."""
    _shape = len(ite)
    _return = np.zeros(_shape, dtype=dtype)
    for i, x in enumerate(ite):
        if typecheck(x):
            _return[i] = x[0]
        else:
            _return[i] = x
    return _return


def _2d(ite, dtype):
    """Create 2d array."""
    _shape = tuple([len(ite), len(ite[0])][::-1])
    _return = np.zeros(_shape, dtype=dtype)
    for i, x in enumerate(ite):
        if typecheck(x):
            for j, y in enumerate(x):
                _return[j, i] = y
        else:
            for j in range(_shape[0]):
                _return[j, i] = x
    return _return


def list_array(ite, dtype=np.float64, verbose=False):
    """Transform list to numpy array of dtype."""
    assert typecheck(ite)
    inner = typecheck(ite[0])
    if inner:
        try:
            _return = _2d(ite, dtype)
        except TypeError as te:
            print(str(te) + '\nNot a 2D array...')
            _return = _1d(ite, dtype)
    else:
        _return = _1d(ite, dtype)
    if verbose:
        print('Converted to shape with:', _return.shape)
    return _return

# end of file
