"""."""

# internal modules

# external modules

# relative modules
from .load import load_cfg
from .functions import _strip
from .colours import FAIL, _RST_, HEADER

# global attributes
__all__ = ('test', 'main')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1

__checks__ = {'.py': '_python',
              '.sh': '_sh'}
"""This attribute should match a corresponding file in
<check_file_template/{}.py> and should also match a function
within this file for parsing."""

def load_func(my_func, *args, **kwargs):
    """Wrapper to access functions."""
    return globals()[my_func](*args, **kwargs)


def _default():
    """."""
    return load_cfg(f'{__path__}check_file_templates/default.py')


def _python():
    """."""
    return load_cfg(f'{__path__}check_file_templates/python.py')


def _sh():
    """."""
    return load_cfg(f'{__path__}check_file_templates/sh.py')


def _parse(parse, row):
    if row.startswith(parse):
        return row.strip(parse)
    return

def _chk_fmt():
    """."""
    pass


def main(fname, chk=2):
    """Main caller function."""
    assert chk in (-1, 1, 2, 3, 4, 5, 6, 7)
    fext = '.' + fname.split('.')[-1]
    if fext not in __checks__.keys():
        return
    else:
        cfg = load_func(__checks__[fext])

    toret = []
    cancel = False

    with open(fname, 'r') as f:
        for row in f:
            # if cancel flag thrown
            if cancel is True:
                break
            # if line is the file escape
            elif row.startswith(cfg.ignore_infile_identifier):
                break
            # -1 grab all and verify matches
            # 1 func check header or file format matches
            # 2 func grab available docstring
            # 3 grab docstring with fname
            # 4 grab all globals
            # 5 grab all functions
            # 6 grab author
            # 7 grab version
            if chk is 1:
                pass
            if chk is 2:
                toret = _parse(cfg.doc_parsing, row)
                if toret:
                    cancel = True
            if (chk is 3) or (chk is -1):
                toret.append(fname)
                toret.append(_parse(cfg.doc_parsing, row))
                if toret:
                    cancel = True
            if (chk is 4) or (chk is -1):
                toret.append(_parse(cfg.globals_parsing, row))
            if (chk is 5) or (chk is -1):
                toret.append(_parse(cfg.function_parsing, row))
            if (chk is 6) or (chk is -1):
                toret.append(_parse(cfg.author_parsing, row))
            if (chk is 7) or (chk is -1):
                toret.append(_parse(cfg.version_parsing, row))
            if chk is -1:
                cancel = False
    if toret is not None:
        if len(toret) == 0:
            print(f'{FAIL}{fname} doesn\'t have parseable lines.{_RST_}')
        return _strip(toret, '\n')
    else:
        print(f'{FAIL}{fname} doesn\'t have parseable lines.{_RST_}')
        return _strip(toret, '\n')


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code
