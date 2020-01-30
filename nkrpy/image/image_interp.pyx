"""Image Interpolation functions."""
# flake8: noqa

# standard modules
import types

# external modules
import numpy as np

# relative modules
from ..misc.functions import typecheck

# global attributes
__all__ = ('interpolate', )
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


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

# end of code

# end of file
