"""Misc Common Functions."""

# internal modules
from operator import attrgetter
from math import cos, sin, acos, ceil
from decimal import Decimal
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


def get(iterable, **attrs):
    """Find first instance of element in iterable.

    A helper that returns the first element in the iterable that meets
    all the traits passed in ``attrs``.
    When multiple attributes are specified, they are checked using
    logical AND, not logical OR. Meaning they have to meet every
    attribute passed in and not one of them.
    To have a nested attribute search (i.e. search by ``x.y``) then
    pass in ``x__y`` as the keyword argument.
    If nothing is found that matches the attributes passed, then
    ``None`` is returned.

    Examples
    ---------
    Basic usage:
        get(members, name='Foo') # searches members.
        __iter__().name for first instance of Foo
    Multiple attribute matching:
        get(channels, name='Foo', bitrate=64000) # searches first instance
           (channels.__iter__().name, ...bitrate) == (Foo, 6400)
    Nested attribute matching:
        get(channels(), guild__name='Cool', name='general') # searches first instance
           (channels.__iter__().guild.name, ...name) == (Cool, general)
    Parameters
    -----------
    iterable
        An iterable to search through.
    **attrs
        Keyword arguments that denote attributes to search with.
    """

    # global -> local
    _all = all
    attrget = attrgetter

    # Special case the single element call
    if len(attrs) == 1:
        k, v = attrs.popitem()
        pred = attrget(k.replace('__', '.'))
        for elem in iterable:
            if pred(elem) == v:
                return elem
        return None

    converted = [
        (attrget(attr.replace('__', '.')), value)
        for attr, value in attrs.items()
    ]

    for elem in iterable:
        if _all(pred(elem) == value for pred, value in converted):
            return elem
    return None


def format_decimal(x, prec=2):
    x = Decimal(str(x))
    tup = x.as_tuple()
    digits = list(tup.digits[:prec + 1])
    sign = '-' if tup.sign else '+'
    dec = ''.join(str(i) for i in digits[1:])
    exp = x.adjusted()
    return f'{sign}{digits[0]}.{dec}D{exp}'


def typecheck(obj):
    """Check if object is iterable (array, list, tuple) and not string."""
    return not isinstance(obj, str) and isinstance(obj, Iterable)


def list_comp(base, comp):
    """Compare 2 lists, make sure purely unique. True if unique"""
    l1 = set(comp)
    l2 = set(base)
    return len(l1.intersection(l2)) == 0


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


def between(l1, lower, upper, exclusive_lower: bool = True, exclusive_upper: bool = True):
    """Find values between bounds, exclusively.

    Return the index and value for everything in iterable
    between v1 and v2, exclusive."""
    if (len(l1) == 0) or (lower == upper):
        return
    if not isinstance(l1, np.ndarray):
        l1 = np.array(l1, dtype=type(l.__next__()))
    if exclusive_lower:
        upper_mask = l1 > lower
    else:
        upper_mask = l1 >= lower
    if exclusive_upper:
        lower_mask = l1 < lower
    else:
        lower_mask = l1 <= upper

    mask = lower_mask and upper_mask

    return mask


def find_nearest(array, value, lower: bool=False, upper: bool=False):
    """Find nearest value within array."""
    if not isinstance(array, np.ndarray):
        array = np.array(array)
    if (not lower and not upper) or (lower and upper):
        idx = (np.abs(array - value)).argmin()
    elif upper:
        temparray = (array - value)
        comp = temparray >= 0
        idx = (temparray[comp]).argmin()
        idx = np.where(array == temparray[comp][idx] + value)[0]
    elif lower:
        temparray = (array - value)
        comp = temparray <= 0
        idx = (np.abs(temparray[comp])).argmin()
        idx = np.where(array == temparray[comp][idx] + value)[0]
    return idx, array[idx]

def find_max(a):
    """Find max of n dimensional array."""
    a = np.array(a)
    shape = len(a.shape())
    if shape == 0:
        return
    maxv = np.amax(a)
    loca = np.array(np.where(a >= maxv))
    return loca, maxv

def test():
    """Testing function for module."""
    pass

if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code
