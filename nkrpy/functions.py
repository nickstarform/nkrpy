"""Misc Common Functions."""

# internal modules
from math import cos, sin, acos, ceil
import os
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable
from IPython import embed

# external modules
import numpy as np
from scipy.optimize import curve_fit

# relative modules

# global attributes
__all__ = ('typecheck', 'addspace', 'between',
           'find_nearest', 'find_nearest_above')
__doc__ = """Just generic functions that I use a good bit."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def typecheck(obj):
    """Check if object is iterable (array, list, tuple) and not string."""
    return not isinstance(obj, str) and isinstance(obj, Iterable)


def list_comp(base, comp):
    """Compare 2 lists, make sure purely unique. True if unique"""
    l1 = set(comp)
    l2 = set(base)
    unique = True
    for x in l1:
        if x in l2:
            unique=False
            break
        else:
            for y in l2:
                if x in y:
                    unique=False
                    break
    return unique


def addspace(arbin, spacing='auto'):
    if typecheck(arbin):
        if str(spacing).lower() == 'auto':
            spacing = max([len(x) for x in map(str, arbin)]) + 1
            return [_add(x, spacing) for x in arbin]
        elif isinstance(spacing, int):
            return [_add(x, spacing) for x in arbin]
    else:
        arbin = str(arbin)
        if str(spacing).lower() == 'auto':
            spacing = len(arbin) + 1
            return _add(arbin, spacing)
        elif isinstance(spacing, int):
            return _add(arbin, spacing)
    raise(TypeError, f'Either input: {arbin} or spacing: {spacing} are of incorrect types. NO OBJECTS')


def _add(sstring, spacing=20):
    """Regular spacing for column formatting."""
    sstring = str(sstring)
    while True:
        if len(sstring) >= spacing:
            sstring = sstring[:-1]
        elif len(sstring) < (spacing - 1):
            sstring = sstring + ' '
        else:
            break
    return sstring + ' '


def _strip(array, var=''):
    """Kind of a quick wrapper for stripping lists."""
    if array is None:
        return
    elif isinstance(array, str):
        return array.strip(var)
    _t = []
    for x in array:
        x = x.strip(' ').strip('\n')
        if var:
            x = x.strip(var)
        if x != '':
            _t.append(x)
    return tuple(_t)


def between(l1, val1, val2):
    """Find values between bounds, exclusively."""
    """Return the value and index for everything in iterable
    between v1 and v2, exclusive."""
    if val1 > val2:
        low = val2
        high = val1
    elif val1 < val2:
        low = val1
        high = val2
    else:
        print('Values are the same')
        return []

    l2 = []
    for j, i in enumerate(l1):
        if(i > low) and (i < high):
            l2.append([j, i])
    return l2


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
        idx = argmin[0]
    return idx, array[idx]

def find_nearest_above(my_array, target):
    if not isinstance(my_array, np.ndarray):
        my_array = np.array(my_array)
    diff = my_array - target
    mask = np.ma.less_equal(diff, 0)
    # We need to mask the negative differences and zero
    # since we are looking for values above
    if np.all(mask):
        return find_nearest(my_array, target) 
    masked_diff = np.ma.masked_array(diff, mask)
    i = masked_diff.argmin()
    if i is None:
        return find_nearest(my_array, target)
    else:
        return i, my_array[i]

def test():
    """Testing function for module."""
    pass

if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code
