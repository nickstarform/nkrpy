"""This code finds the jacobi constant from given files."""

# standard modules
import os
from sys import version

# external modules

# relative Modules
from .error import ConfigError
from .decorators import deprecated

__version__ = float(version[0:3])
__cpath__ = '/'.join(os.path.realpath(__file__).split('/')[:-1])

__cwd__ = os.getcwd()


def verify_dir(name):
    """Wrapper for directories."""
    if not os.path.isdir(name):
        os.mkdir(name)


def verify(target, comparison):
    """Verify parameters within file against a template."""
    if isinstance(target, str):
        target = load_cfg(target)
    raise ConfigError


def load_cfg(fname):
    """Load configuration file as module."""
    if '/' not in fname:
        fname = __cwd__ + '/' + fname
    try:
        if __version__ >= 3.5:
            import importlib.util
            spec = importlib.util.spec_from_file_location("config", fname)
            cf = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(cf)
        elif __version__ >= 3.3:
            from importlib.machinery import SourceFileLoader
            cf = SourceFileLoader("config", fname).load_module()
        elif __version__ <= 3.0:
            import imp
            cf = imp.load_source('config', fname)
    except:
        print('Failed. Cannot find file <{}> or the fallback <config.py>'
              .format(fname))
        print('Or invalid line found in file. Try using import <{}> yourself'
              .format(fname[:-3]))
        exit(1)
    return cf


@deprecated
def load_variables(mod):
    """Given a module, load attributes directly to global."""
    for k in dir(mod):
        if '__' not in k:
            globals()[k] = getattr(mod, k)

if __name__ == '__main__':
    print('Testing')

# end of file
