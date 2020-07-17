"""This code manipulates loading of files."""

# standard modules
import os
from sys import version

# external modules

# relative Modules
from ..misc.errors import ConfigError
from ..misc.decorators import deprecated
from ..misc.colours import FAIL, _RST_, HEADER
from ..misc.functions import typecheck

# global attributes
__all__ = ('verify_param', 'verify_dir', 'load_cfg')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = float(version[0:3])
__cwd__ = os.getcwd()
__version__ = float(version[0:3])


def verify_dir(name, create=False):
    """Verify/create a directory."""
    if not os.path.isdir(name):
        if create:
            os.makedirs(name)
        else:
            return False
    return True


def verify_param(target, comparison):
    """Verify parameters within file against a template."""
    """All of comparison must be in target, not vice versa."""
    if isinstance(target, str):
        target = load_cfg(target)
    if isinstance(comparison, str):
        comparison = load_cfg(comparison)
    if isinstance(target, dict):
        _t = [x for x in target.keys() if '__' not in x]
    elif typecheck(target):
        _t = [x for x in target if '__' not in x]
    else:
        _t = [x for x in dir(target) if '__' not in x]
    if isinstance(comparison, dict):
        _c = [x for x in comparison.keys() if '__' not in x]
    elif typecheck(comparison):
        _c = [x for x in comparison if '__' not in x]
    else:
        _c = [x for x in dir(comparison) if '__' not in x]

    for x in _c:
        if x not in _t:
            raise ConfigError(f'{FAIL}Parameters not found.{_RST_}',  # noqa
                              (target, comparison))
    else:
        return True


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
    except Exception:
        print(f'{FAIL}Failed.{_RST_}Cannot find file <{HEADER}{fname}{_RST_}> or the fallback config.py>')  # noqa
        print(f'Or invalid line found in file. Try using import <{HEADER}{fname[:-3]}{_RST_}> yourself')  # noqa
        exit(1)
    return cf


@deprecated
def load_variables(mod):
    """Given a module, load attributes directly to global."""
    for k in dir(mod):
        if '__' not in k:
            globals()[k] = getattr(mod, k)


@deprecated
def load_func(my_func, *args, **kwargs):
    """Access functions."""
    return locals()[my_func](*args, **kwargs)


if __name__ == '__main__':
    print('Testing')

# end of file
