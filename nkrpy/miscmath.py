"""Various math functions."""

import numpy as np
from .constants import pi
from math import ceil, cos, sin, acos


def linear(x, a, b):
    """Linear function."""
    return a * x + b


def binning(data, width=3):
    """Bin the given data."""
    return data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)


def find_nearest(array, value):
    """Find nearestvalue within array."""
    if isinstance(array, np.ndarray):
        idx = (np.abs(array - value)).argmin()
    else:
        argmin = (float('inf'), float('inf'))
        for i, x in enumerate(array):
            _tmp = np.abs(value - x)
            if _tmp < argmin[1]:
                argmin = (i, _tmp)
        idx = i
    return idx, array[idx]


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


def dot_product(a, b):
    """Compute dot product between a and b."""
    ret = 0
    for i in range(len(a)):
        ret += a[i] * b[i]
    return ret


def mag(a):
    """Mag of a vector."""
    ret = 0
    for x in a:
        ret += x ** 2
    return (ret) ** 0.5


def length(*args):
    """Same as mag."""
    return mag(*args)


def ang_vec(deg):
    """Unit vec of deg."""
    rad = deg * pi / 180.
    return cos(rad), sin(rad)


def determinant(v, w):
    """Determinant of two vecs."""
    return v[0] * w[1] - v[1] * w[0]


def inner_angle(v, w):
    """Calculate inner angle between two vecs."""
    cosx = dot_product(v, w) / (length(v) * length(w))
    while np.abs(cosx) >= 1:
        cosx = cosx / (np.abs(cosx) * 1.001)
    rad = acos(cosx)  # in radians
    return rad * 180. / pi  # returns degrees


def angle_clockwise(A, B):
    """Determine the angle clockwise between vecs."""
    inner = inner_angle(A, B)
    det = determinant(A, B)
    if det > 0:  # this is a property of the det.
        # If the det < 0 then B is clockwise of A
        return inner
    else:  # if the det > 0 then A is immediately clockwise of B
        return 360. - inner


def gen_angles(start, end, resolution=1, direction='+'):
    """Generate angles between two designations."""
    if direction == "+":  # positive direction
        diff = round(angle_clockwise(ang_vec(start), ang_vec(end)), 2)
        numd = int(ceil(diff / resolution)) + 1
        final = [round((start + x * resolution), 2) % 360 for x in range(numd)]
    elif direction == "-":  # negative direction
        diff = round(360. - angle_clockwise(ang_vec(start), ang_vec(end)), 2)
        numd = int(ceil(diff / resolution)) + 1
        final = [round((start - x * resolution), 2) % 360 for x in range(numd)]
    return final


def gauss(x, mu, sigma, a):
    """Define a single gaussian."""
    return a * np.exp(-(x - mu) ** 2 / 2. / sigma**2)


def ndgauss(x, params):
    """N dimensional gaussian."""
    """
    assumes params is a 2d list
    """
    for i, dim in enumerate(params):
        if i == 0:
            final = gauss(x, *dim)
        else:
            final = np.sum(gauss(x, *dim), final, axis=0)

    return final


def addconst(func, c):
    """Add constant."""
    return func + c


def polynomial(x, params):
    """Polynomial function."""
    """
    assuming params is a 1d list
    constant + 1st order + 2nd order + ...
    """
    for i, dim in enumerate(params):
        if i == 0:
            final = [dim for y in x]
        else:
            final = np.sum(dim * x ** i, final, axis=0)

    return final


def baseline(x, y, order=2):
    """Fit a baseline."""
    """
    Input the xvals and yvals for the baseline
    Will return the function that describes the fit
    """
    fit = np.polyfit(x, y, order)
    fit_fn = np.poly1d(fit)
    return fit_fn


def listinvert(total, msk_array):
    """Invert list with mask."""
    """
    msk_array must be the index values
    """
    mask_inv = []
    for i in range(len(msk_array)):
        mask_inv = np.append(mask_inv, np.where(total == msk_array[i]))
    mask_tot = np.linspace(0, len(total) - 1, num=len(total))
    mask = np.delete(mask_tot, mask_inv)
    mask = [int(x) for x in mask]
    return mask

# end of file
