"""."""
from . import colours
from . import constants
from . import decorators
from . import errors
from . import mp
from . import functions
from .frozendict import FrozenOrderedDict
from .frozendict import FrozenDict # noqa

__all__ = ['FrozenDict', 'FrozenOrderedDict', 'mp', 'colours', 'constants', 'decorators', 'errors', 'functions']


PACKAGES = __all__.copy()
PACKAGES.sort()

# end of code

# end of file
