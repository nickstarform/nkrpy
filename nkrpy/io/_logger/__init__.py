from .logger import *  # noqa

__doc__ = Log.__doc__  # noqa

# alias
Logger = Log

__all__ = ['Log', 'Logger']

PACKAGES = [a for a in dir(Log) if not a.startswith('__')]
PACKAGES.sort()

# end of code

# end of file
