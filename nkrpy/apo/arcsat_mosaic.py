"""Generator for ARCSAT mosaics."""

# internal modules
from argparse import ArgumentParser as ap

# external modules
import numpy as np

# relative modules
from ..miscmath import raster_matrix

# global attributes
__all__ = ('test',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__name__)
__version__ = 0.1


def main():
    """Main caller function."""
    pass


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""
    parser = ap('Generator for ARCSAT mosaics.')

    print('Testing module')
    test()
    print('Test Passed')

# end of code
