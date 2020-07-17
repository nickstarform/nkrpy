"""Various math functions."""
# cython modules

# standard modules
from itertools import chain, islice, groupby
from operator import itemgetter

# external modules
import numpy as np

# relative modules
from ..misc.constants import pi
from ..misc.functions import typecheck

# global attributes
__all__ = ('flatten', 'listinvert', 'binning', 'cross',
           'dot', 'radians', 'deg', 'mag',
           'ang_vec', 'determinant', 'inner_angle',
           'angle_clockwise', 'apply_window', 'list_array',
           'pairwise', 'window', 'group_ranges',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def group_ranges(data):
    """Yield range of consecutive numbers."""
    for k, g in groupby(enumerate(data), lambda x: x[0] - x[1]):
        group = (map(itemgetter(1), g))
        group = list(map(int, group))
        if len(group) == 0:
            yield (group[0], )
        yield (group[0], group[-1])

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
        ind = np.arange(0, points.shape[0]).tolist()
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


def flatten(inputs) -> list:
    """."""
    ret = list(chain.from_iterable(inputs))
    return ret


def binning(data, width=3):
    """Bin the given data."""
    if width == 1:
        return data[:]
    return data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)


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


def apply_window(ilist: list, window: float):
    """Apply a given constant window to a list.

    Parameters
    ----------
    ilist: list
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
    ret = []
    i = 0
    current = ilist[0]
    tmp = []
    while i < len(ilist):
        if (ilist[i] - current) < window:
            tmp.append(ilist[i])
        elif (i <= len(ilist) - 1):
            ret.append(sum(tmp) / len(tmp))
            tmp = [ilist[i]]
            current = ilist[i]
        i += 1
        if i == len(ilist):
            tmp.append(ilist[i - 1])
            ret.append(sum(tmp) / len(tmp))

    return ret


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
