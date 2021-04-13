"""Generalized type casting.

This is an effort to normalize al the custom classes within `nkrpy`
so that type() and isinstance() produces results that are natural
and simple. These classes implement no methods and are just used
for typing.
"""
# cython modules

# internal modules

# external modules

# relative modules

# global attributes
__all__ = ['WCSClass', 'UnitClass', 'FileClass', 'LoggerClass', 'PlotClass']
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


class PlotClass(object):
    """Logger class for nkrpy.io.Logger."""

    pass


class LoggerClass(object):
    """Logger class for nkrpy.io.Logger."""

    pass


class FileClass(object):
    """File class for nkrpy.io."""

    pass


class WCSClass(object):
    """WCS class for nkrpy.astro.WCS."""

    pass


class UnitClass(object):
    """Unit class for nkrpy.Unit."""

    pass

# end of code

# end of file
