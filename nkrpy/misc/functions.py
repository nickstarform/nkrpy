"""Misc Common Functions."""

# internal modules
from operator import attrgetter, add
from collections.abc import Iterable
import inspect
from collections import Mapping
from itertools import chain
from functools import reduce

# external modules
import numpy as np

# relative modules

# global attributes
__all__ = ['typecheck', 'flatten', 'addspace', 'between',
           'find_nearest', 'strip', 'get', 'list_comp',
           'add', 'find_max', 'deep_resolve', 'help', 'help_api', 'flatten_dict', 'deep_get_single', 'deep_get_generator', 'deep_set', 'classdocgen', 'cli', 'FunctionalFunction']
__doc__ = """Just generic functions that I use a good bit."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__LINEWIDTH__ = 79


def flatten(ls):
    for items in ls:
        if typecheck(items):
            for subitem in flatten(items):
                yield subitem
            continue
        yield items



class FunctionalFunction(object):
    pass
    """
    def __init__(self, func=lambda x: 0):
            self.func = func

    def __call__(self, *args, **kwargs):
            return self.func(*args, **kwargs)

    def __add__(self, other):
        if isinstance(other, FunctionalFunction):
            return FunctionalFunction(lambda *args; self.func(*args) + other.func(*args))
        return self + FunctionalFunction(other)

    def __mul__(self, other):
            def composed(*args, **kwargs):
                    return self(other(*args, **kwargs))
            return composed
"""


def dict_split(kwarg, tosplit: list = []):
    """Split dict based list of keys."""
    cut_kwarg = {}
    ret = {}
    for k in kwarg:
        if k in tosplit:
            ret[k] = kwarg[k]
        else:
            cut_kwarg[k] = kwarg[k]
    return cut_kwarg, ret



def classdocgen(clas):
    def grab_first_line(func):
        if func.__doc__ is None:
            return ''
        doc = func.__doc__.splitlines()
        if len(doc) == 0:
            return ''
        return doc[0]

    methods = [f'{f}: {grab_first_line(getattr(clas, f))}' for f in dir(clas) if not f.startswith('_') and callable(getattr(clas, f))]
    spacer = '''
'''
    return spacer.join(methods)


def __get(dct, key, default):
    return reduce(lambda d, k: d.get(k, default) if isinstance(d, dict) else default, key, dct)


def deep_get_generator(dct: dict, *keys, default=None, delim: str = '.'):
    """Resolve keys by splitting the list via '.'.

    Usage
    =====
    test = {
        'a': 1,
        'b': {
            'aa': 11,
            'bb': {
                'aaa': 111,
            },
        },
    }
    deep_get(test, 'a', 'b.aa', 'b.bb.aaa', 'b.bb.bbb')
    # return 1, 11, 111, None
    """
    for key in keys:
        key = key.split(delim)
        yield __get(dct, key, default)

def deep_get_single(dct: dict, key, default=None, delim: str = '.'):
    """Resolve keys by splitting the list via '.'.

    Usage
    =====
    test = {
        'a': 1,
        'b': {
            'aa': 11,
            'bb': {
                'aaa': 111,
            },
        },
    }
    deep_get(test, 'a', 'b.aa', 'b.bb.aaa', 'b.bb.bbb')
    # return 1, 11, 111, None
    """
    key = key.split(delim) if isinstance(key, str) else key
    return __get(dct, key, default)

def deep_set(dct, key, value, delim: str = '.'):
    """Set key by splitting the str via delim.

    Usage
    =====
    test = {
        'a': 1,
        'b': {
            'aa': 11,
            'bb': {
                'aaa': 111,
            },
        },
    }
    111 == test['b']['bb']['aaa'] # True
    deep_set(test, 'b.bb.aaa', 333)
    111 == test['b']['bb']['aaa'] # False
    333 == test['b']['bb']['aaa'] # True
    # 
    """
    keys = key.split(delim)
    i = 0
    while i < len(keys) - 1:
        k = keys[i]
        i += 1
        dct = dct.get(k)
    dct[keys[-1]] = value


def flatten_dict(d, join=add, lift=lambda x: x, only_keys: bool = False):
    """Flatten a dictionary intelligently.

    This is a little complicated

    Usage
    =====
    from nkrpy.misc.functions import flatten_dict
    test = {
        'a': 1,
        'b': {
            'aa': 11,
            'bb': {
                'aaa': 111,
            },
        },
    }
    flatten_dict(test, lift=lambda x: (x,), only_keys=True)

    Parameters
    ==========
    d: dict
        The dictionary to flatten
    join: function
        The method by which to stack the 'lifts'
    lift: function
        A lambda function by which to transform the keys
    only_keys: bool
        Flag to only return the keys or return a full dupe

    """
    _FLAG_FIRST = object()
    results = []
    def visit(sd, r, partKey):
        for key, value in sd.items():
            nk = lift(key) if partKey is _FLAG_FIRST else join(partKey, lift(key))
            if isinstance(value, Mapping):
                visit(value, r, nk)
            else:
                if not only_keys:
                    r.append((nk, value))
                else:
                    r.append(nk)
    visit(d, results, _FLAG_FIRST)
    return results


def deep_resolve(iterable):
    """Deep resolver to resolve the lowest possible element of an iterable."""
    if not isinstance(iterable, Iterable) or isinstance(iterable, str):
        yield (iterable, )
    try:
        for sub in iterable:
            for element in deep_resolve(sub):
                yield element
    except TypeError:
        yield (iterable, )


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
           (channels.__iter__().name, bitrate) == (Foo, 6400)
    Nested attribute matching:
        get(channels(), guild__name='Cool', name='general') # first instance
           (channels.__iter__().guild.name, name) == (Cool, general)
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


def typecheck(obj):
    """Check if object is iterable (array, list, tuple) and not string."""
    return not isinstance(obj, str) and isinstance(obj, Iterable)


def list_comp(base, comp):
    """Compare 2 lists, make sure purely unique.

    True if unique
    """
    l1 = set(comp)
    l2 = set(base)
    return len(l1.intersection(l2)) == 0


def addspace(arbin, spacing: int = 'auto'):
    """Add space to end of (list) of strings."""
    if typecheck(arbin):
        if str(spacing).lower() == 'auto':
            spacing = max([len(x) for x in map(str, arbin)]) + 1
            return [add(x, spacing) for x in arbin]
        elif isinstance(spacing, int):
            return [add(x, spacing) for x in arbin]
    else:
        arbin = str(arbin)
        if str(spacing).lower() == 'auto':
            spacing = len(arbin) + 1
            return add(arbin, spacing)
        elif isinstance(spacing, int):
            return add(arbin, spacing)
    raise(TypeError, f'Either input: {arbin} or spacing:' +
                     f'{spacing} are of incorrect types. NO OBJECTS')


def add(sstring: str, spacing: int = 20):
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


def strip(array, var=''):
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


def between(l1, lower, upper, exclusive_lower: bool = True,
            exclusive_upper: bool = True):
    """Find values between bounds, exclusively.

    Return the index and value for everything in iterable
    between v1 and v2, exclusive.
    """
    if (len(l1) == 0) or (lower == upper):
        return
    if not isinstance(l1, np.ndarray):
        l1 = np.array(l1, dtype=type(l1.__next__()))
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


def find_nearest(array, value, lower: bool = False, upper: bool = False):
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


# end of code
