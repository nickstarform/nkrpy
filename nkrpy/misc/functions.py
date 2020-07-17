"""Misc Common Functions."""

# internal modules
from operator import attrgetter
from collections.abc import Iterable
import inspect

# external modules
import numpy as np

# relative modules
from . import colours

# global attributes
__all__ = ('typecheck', 'addspace', 'between',
           'find_nearest', 'strip', 'get', 'list_comp',
           'add', 'find_max', 'deep_resolve', 'help', 'help_api')
__doc__ = """Just generic functions that I use a good bit."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__LINEWIDTH__ = 79


def help(func_or_class, colour: bool = True):
    ret = help_api(func_or_class, colour)
    print(ret)


def help_api(func_or_class, colour: bool = False):
    name = func_or_class.__name__
    top_level_docs = func_or_class.__doc__ if func_or_class.__doc__ else ''
    top_level_mod = func_or_class.__module__ if '__module__' in dir(func_or_class) else ''
    name = (f'{top_level_mod}.{name}').upper()
    ret = ''
    if '__class__' not in dir(func_or_class):
        args = str(inspect.signature(func_or_class))
    else:
        dirs = [d for d in dir(func_or_class) if not d.startswith('_')]
        dirs = [getattr(func_or_class, d) for d in dirs]
        args = ''
        for d in dirs:
            inner_name = d.__name__
            inner_docs = d.__doc__ if d.__doc__ else ''
            inspected_inner = inspect.signature(d).parameters
            inner_args = ', '.join([inspected_inner for i, v in enumerate(inspected_inner) if i > 0])
            toplevel = f'{inner_name}: {inner_args}'
            args += toplevel + '\n'
            if len(inner_docs) > 20:
                args += ('-' * len(toplevel)) + '\n'
                args += inner_docs + '\n'
            else:
                args += f': {inner_docs}' + '\n'
    if colour:
        ret += colours.HEADER + colours._BLD
    spacer = ' ' * int(__LINEWIDTH__ / 2 - len(name) / 2 - 1)
    ret += ('=' * __LINEWIDTH__) + '\n'
    ret += spacer + name + spacer + '\n'
    ret += ('=' * __LINEWIDTH__) + '\n'
    if colour:
        ret += colours._RST_ + colours.OKBLUE
    ret += top_level_docs + '\n'
    ret += ('-' * __LINEWIDTH__) + '\n'
    if colour:
        ret += colours.WARNING
    ret += args
    if colour:
        ret += colours._RST_
    return ret


def deep_resolve(iterable):
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
           (channels.__iter__().name, ...bitrate) == (Foo, 6400)
    Nested attribute matching:
        get(channels(), guild__name='Cool', name='general') # first instance
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
