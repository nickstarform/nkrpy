"""."""

from . import _config
from ._config import *
from . import _stdio
from ._stdio import *
from . import _files
from ._files import *
from . import _fits
from ._fits import *
from . import _logger
from ._logger import *
from . import _sorting
from ._sorting import *
from . import _sizeof
from ._sizeof import *
from . import template

__all__ = ['template'] + \
          _config.__all__ +\
          _stdio.__all__ +\
          _files.__all__ +\
          _fits.__all__ +\
          _logger.__all__ +\
          _sorting.__all__ +\
          _sizeof.__all__

PACKAGES = __all__.copy()
PACKAGES.sort()

# end of code

# end of file
