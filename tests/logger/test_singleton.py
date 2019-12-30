"""."""
# flake8: noqa

# internal modules

# external modules

# relative modules
from nkrpy import logger

# global attributes
__all__ = ('singleton_helper',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def singleton_helper():
    logger.set_verbosity(logger.get_verbosity() % 5)

# end of code

# end of file
