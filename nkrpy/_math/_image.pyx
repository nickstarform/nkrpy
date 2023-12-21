"""."""
# flake8: noqa
# cython modules
cimport numpy as cnp
cimport cython

# internal modules

# external modules
import numpy as np
from cython.parallel import prange as crange
from skimage.transform import rotate as sk__rotate

# relative modules
from .miscmath import angle_clockwise, ang_vec
from ..misc import typecheck

# global attributes
__all__ = ['raster_matrix', 'gen_angles', 'rotate_points',
           'rotate_matrix', 'shift', 'rotate']
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.
cpdef shift(double[:, :, :] data: np.ndarray, long[:] shifts: np.ndarray,
            bint override: bool = True, float cval: float = np.NaN):
    """Shift a matrix.

    Shift a matrix by a certain amount. Assumed input size is 3D, those axis 
    can be blank, just make sure the corresponding shifts are 0.

    Parameters
    ----------
    data: np.ndarray
        The data to shift.
    shifts: np.ndarray
        Array of shifts to perform, must correspond to same axis as data.
    override: bool
        If set to true, will override initial array.
            Is slower but memory efficient
    """
    cdef double[:, :, :] shifted = np.roll(data, shifts,
                                           axis=list(range(shifts.shape[0])))
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0
    if not override:
        shifted[:shifts[0], ...] = cval
        shifted[:, :shifts[1], :] = cval
        shifted[..., :shifts[2]] = cval
        return np.asarray(shifted)
    else:
        for i in range(shifted.shape[0]):
            for j in range(shifted.shape[1]):
                for k in crange(shifted.shape[2], nogil=True):
                    if i < shifts[0] or j < shifts[1] or k < shifts[2]:
                        data[i, j, k] = cval
                        continue
                    data[i, j, k] = shifted[i, j, k]


def rotate(double[:, :] data: np.ndarray,
           long angle, float cval: float = np.NaN):
    """Rotate image by a certain angle around its center.
    Parameters
    ----------
    image : ndarray
        Input image.
    angle : float
        Rotation angle in degrees in counter-clockwise direction.
    Returns
    -------
    rotated : ndarray
        Rotated version of the input.
    
    """
    return sk__rotate(data, angle, resize=False, center=None, order=1,
                      mode='constant', cval=cval)


"""
https://math.stackexchange.com/questions/2004800/math-for-simple-3d-coordinate-rotation-python
new_yaxis = -np.cross(new_xaxis, new_zaxis)

# new axes:
nnx, nny, nnz = new_xaxis, new_yaxis, new_zaxis
# old axes:
nox, noy, noz = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=float).reshape(3, -1)

# ulgiest rotation matrix you can imagine
top = [np.dot(nnx, n) for n in [nox, noy, noz]]
mid = [np.dot(nny, n) for n in [nox, noy, noz]]
bot = [np.dot(nnz, n) for n in [nox, noy, noz]]

def newit(vec):
    xn = sum([p*q for p,q in zip(top, vec)])
    yn = sum([p*q for p,q in zip(mid, vec)])
    zn = sum([p*q for p,q in zip(bot, vec)])
    return np.hstack((xn, yn, zn))
"""

def interpolate(obj, val, dtype='linear'):
    '''
    obj can be any iterable form of 1d or 2d. Easily transferable to higher D
    val is the index for interpolate over
    dtype is the interpolation technique
    '''
    # handle obj types
    if type(obj) == np.ndarray:
        shapeO = obj.shape
    else:
        shapeO = (len(obj),)
    # handle val types
    if type(val) == np.ndarray:
        shapeV = val.shape
    elif typecheck(val):
        shapeV = (len(val), )
    else:
        val = [val, ]
        shapeV = (len(val),)

    def left(obj, val):
        '''
        assuming singular value
        '''
        if val == 0:
            return False
        return obj[int(val) - 1]

    def right(obj, shapeO, val):
        '''
        assuming singular value
        '''
        if val == -1:
            return False
        elif val == (shapeO[0] - 1):
            return False
        return obj[val + 1]

    def upper(obj, shapeO, val):
        '''
        assuming val is 2D now
        '''
        if len(shapeO) == 1:
            return False
        elif val[1] >= shapeO[1]:
            return False
        return obj[val[0], val[1] + 1]

    def lower(obj, shapeO, val):
        '''
        assuming val is 2D now
        '''
        if len(shapeO) == 1:
            return False
        elif val[1] <= 0:
            return False
        return obj[val[0], val[1] - 1]

    def oneD(obj, shapeO, val):
        for i in val:
            l, r = left(obj, i), right(obj, shapeO, i)
            if l and r:
                obj[i] = interpolate(l, r, obj)
            elif l:
                obj[i] = interpolate(l, obj)
            elif r:
                obj[i] = interpolate(r, obj)

    def twoD(obj, shapeO, val):
        for i in val:
            le, r, u, lo = (left(obj[i[0], :], i[1]),
                            right(obj[i[0], :], shapeO, i[1]),
                            upper(obj[:, i[1]], shapeO, i[0]),
                            lower(obj[:, i[1]], shapeO, i[0]))
            if le and r and u and lo:
                horiz = interpolate(le, r, obj)
                vert  = interpolate(u, lo, obj)
                obj[val] = np.average(horiz, vert)
            elif le:
                obj[i] = interpolate(le, obj)
            elif r:
                obj[i] = interpolate(r, obj)


def raster_matrix(*args, auto=False, **kwargs):
    """Return a matrix of a raster track.

    Assuming all units are the same!

    Parameters
    ----------
    Cen: iterable[float]
        center points
    width: float
        total width (evenly split)
    height: float
        total height (evenly split)
    fov: float
        field of view of window
    auto: bool
        if auto is set will construct a double grid, one of specified
    direction and the next grid of opposite to maximize sensitivity
    main: <h/v> determine which is the major track
    h: str [+ | -]
        direction of starting horizontal track
    v: str  [+ | -]
        direction of starting vertical track
    sample: float
        Amount of overlap. > 0 with 1 being exactly no overlap and
        infinity being complete overlap.

    Usage
    -----
    def _plot_raster(matrix):
        '''Plotter for the raster matrix.'''
        plt.figure(figsize=[16, 16])

        plt.plot(matrix[:, 0], matrix[:, 1], 'r-')
        plt.plot(matrix[:, 0], matrix[:, 1], 'b.')
        plt.plot(matrix[0, 0], matrix[0, 1], '*', color='black', label='start')
        plt.plot(matrix[-1, 0], matrix[-1, 1], '*', color='purple', label='end')
        a = plt.legend()
        plt.title(f'Raster Scan: {matrix[0]} to {matrix[-1]}')
        plt.show()

    """
    if auto:
        firstn, firstm = _raster_matrix_con(*args, **kwargs)
        secondn, secondm = _raster_matrix_con(*args, rev=True, **kwargs)
        totaln = firstn + secondn
        totalm = np.concatenate((firstm, secondm))
    else:
        totaln, totalm = _raster_matrix_con(*args, **kwargs)
    return totaln, totalm


def gen_angles(start, end, resolution=1, direction='+'):
    """Generate angles between two designations."""
    if direction == "+":  # positive direction
        diff = round(angle_clockwise(ang_vec(start), ang_vec(end)), 2)
        numd = int(np.ceil(diff / resolution)) + 1
        final = [round((start + x * resolution), 2) % 360 for x in range(numd)]
    elif direction == "-":  # negative direction
        diff = round(360. - angle_clockwise(ang_vec(start), ang_vec(end)), 2)
        numd = int(np.ceil(diff / resolution)) + 1
        final = [round((start - x * resolution), 2) % 360 for x in range(numd)]
    return final


def rotate_points(origin, point, angle):
    """Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy


def rotate_matrix(origin, matrix, angle):
    """Rotate a matrix counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = matrix[:, 0], matrix[:, 1]

    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return np.concatenate((qx.reshape(qx.shape[0], 1),qy.reshape(qy.shape[0], 1)), axis=1)


def _raster_matrix_con(fov, cen=(0, 0), width=1, height=1, main='h',
                       theta=0, h='+', v='-', sample=2.,
                       box_bounds=None, rev=False):
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
        alldegrees = rotate_matrix(cen, alldegrees, theta)
    return num_centers_w * num_centers_h, alldegrees

# end of code

# end of file
