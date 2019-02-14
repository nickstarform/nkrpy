"""Various math functions."""

# standard modules
from math import ceil, cos, sin, acos
from copy import deepcopy

# external modules
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import rv_continuous

# relative modules
from .constants import pi
from .functions import typecheck

# global attributes
__all__ = ('',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def _raster_matrix_con(fov, cen=(0, 0), width=1, height=1, main='h', theta=0, h='+',
                       v='-', sample=2., box_bounds=None, rev=False, plot=False):
    """Main constructor."""
    if rev:
        h = '+' if h == '-' else '-'
        v = '+' if v == '-' else '-'
        main = 'h' if main == 'v' else 'v'
    print(h, v)
    if not box_bounds:
        box_bounds = ((cen[0] - width / 2., cen[1] + height / 2.),
                      (cen[0] + width / 2., cen[1] + height / 2.),
                      (cen[0] - width / 2., cen[1] - height / 2.),
                      (cen[0] + width / 2., cen[1] - height / 2.))

    num_centers_w = int(np.ceil(2. * width / (fov / sample) - 4))
    num_centers_h = int(np.ceil(2. * height / (fov / sample) - 4))
    vertrange = np.linspace(box_bounds[2][1], box_bounds[0][1], endpoint=True,
                            num=num_centers_h)
    horirange = np.linspace(box_bounds[0][0], box_bounds[1][0], endpoint=True,
                            num=num_centers_w)

    if v == '-':
        vertrange = np.array(list(vertrange)[::-1])
    if h == '-':
        horirange = np.array(list(horirange)[::-1])

    alldegrees = []
    count = 0
    if main == 'h':
        for i, v in enumerate(vertrange):
            _t = list(horirange)

            if count % 2 == 1:  # negative direction
                _t = _t[::-1]

            for x in _t:
                alldegrees.append(np.array([x, v]))
            count += 1
    else:
        for i, h in enumerate(horirange):
            _t = list(vertrange)

            if count % 2 == 1:  # negative direction
                _t = _t[::-1]

            for x in _t:
                alldegrees.append(np.array([h, x]))
            count += 1
    alldegrees = np.array(alldegrees)
    if theta % 360. != 0.:
        _t = deepcopy(alldegrees)
        alldegrees = rotate_matrix(cen, _t, theta)

    if plot:
        _plot_raster(alldegrees)
    return num_centers_w * num_centers_h, alldegrees


def _plot_raster(matrix):
    from IPython import embed
    """Plotter for the raster matrix."""
    plt.figure(figsize=[16, 16])

    plt.plot(matrix[:, 0], matrix[:, 1], 'r-')
    plt.plot(matrix[:, 0], matrix[:, 1], 'b.')
    plt.plot(matrix[0, 0], matrix[0, 1], '*', color='black', label='start')
    plt.plot(matrix[-1, 0], matrix[-1, 1], '*', color='purple', label='end')
    a = plt.legend()
    embed()
    plt.title(f'Raster Scan: {matrix[0]} to {matrix[-1]}')
    plt.draw()
    plt.show()

def raster_matrix(*args, auto=False, **kwargs):
    """Return a matrix of a raster track."""
    """Assuming all units are the same!
    Cen: <iterable[float]> of the center points
    width: <float> of total width (evenly split)
    height: <float> of total height (evenly split)
    fov: <float> field of view of window
    auto: if auto is set will construct a double grid, one of specified
    plot: <boolean> if should plot the output
    direction and the next grid of opposite to maximize sensitivity
    main: <h/v> determine which is the major track
    h: <+/-> for direction of starting horizontal track
    v: <+/-> for direciton of starting vertical track
    sample: <float> give float >0 with 1 being exactly no overlap and
    infinity being complete overlap."""
    if auto:
        firstn, firstm = _raster_matrix_con(*args, **kwargs)
        secondn, secondm = _raster_matrix_con(*args, rev=True, **kwargs)
        totaln = firstn + secondn
        totalm = np.concatenate((firstm, secondm))
    else:
        totaln, totalm = _raster_matrix_con(*args, **kwargs)
    return totaln, totalm


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


def _list_array(ite, dtype=np.float64, verbose=False):
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


def gaussian_sample(lower_bound, upper_bound, size=100, scale=None):
    """Sample from a gaussian given limits."""
    if lower_bound == upper_bound:
        scale = 0
    loc = (lower_bound + upper_bound) / 2.
    if scale is None:
        scale = (upper_bound - lower_bound) / 2.
    results = []
    while len(results) < size:
        samples = np.random.normal(loc=loc, scale=scale,
                                   size=size - len(results))
        results += [sample for sample in samples
                    if lower_bound <= sample <= upper_bound]
    return results


def plummer_density(x, mass, a):
    """Return Plummer density giving cirical radius and mass)."""
    return 3. * mass / (4. * np.pi * a ** 3) * (1. + (x / a) ** 2)**(-5. / 2.)


def plummer_mass(x, mass, a):
    """Return the mass for a plummer sphere."""
    return mass * x ** 3 / ((x ** 2 + a ** 2) ** (3. / 2.))


def linear(x, a, b):
    """Linear function."""
    return a * x + b


def quad(x, a, b, c):
    """Linear function."""
    return a * x ** 2 + b * x + c


def binning(data, width=3):
    """Bin the given data."""
    return data[:(data.size // width) * width].reshape(-1, width).mean(axis=1)


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


def angle_clockwise(a, b):
    """Determine the angle clockwise between vecs."""
    inner = inner_angle(a, b)
    det = determinant(a, b)
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


def rotate_points(origin, point, angle):
    """Rotate a point counterclockwise by a given angle around a given origin."""
    """
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def rotate_matrix(origin, matrix, angle):
    """Rotate a matrix counterclockwise by a given angle around a given origin."""
    """
    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = matrix[:, 0], matrix[:, 1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.concatenate((qx.reshape(qx.shape[0], 1),qy.reshape(qy.shape[0], 1)), axis=1)

# end of file
