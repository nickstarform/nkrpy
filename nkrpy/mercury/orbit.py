"""."""

# internal modules

# external modules
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# relative modules
from ..load import load_config
from .file_loader import parse_aei

# global attributes
__all__ = ('test',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def thumbnail(x, y, z, name):
    pass

def sim_orbit():


def plot_orbit():



def main(cfgname):
    cfg = load_config(cfgname)
    bodies = [parse_aei(x) for x in cfg['files']]



def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code
