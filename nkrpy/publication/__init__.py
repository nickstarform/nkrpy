"""."""
from . import cmaps
from . import _format
from ._format import *
from . import _plots
from ._plots import *

__all__ = ['cmaps'] +\
          _format.__all__ +\
          _plots.__all__


PACKAGES = __all__.copy()
PACKAGES.sort()

# end of code

# end of file
