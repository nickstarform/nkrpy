"""."""

# internal modules

# external modules

# relative modules
from .load import load_cfg
from ..misc.functions import strip
from ..misc.colours import FAIL, _RST_

# global attributes
__all__ = ('main',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

__checks__ = {'.py': '_python',
              '.sh': '_sh'}
"""This attribute should match a corresponding file in
<check_file_template/{}.py> and should also match a function
within this file for parsing."""


def __load_func(my_func, *args, **kwargs):
    """Load function."""
    return globals()[my_func](*args, **kwargs)


def __default():
    """."""
    return load_cfg(f'{__path__}check_file_templates/default.py')


def __python():
    """."""
    return load_cfg(f'{__path__}check_file_templates/python.py')


def __sh():
    """."""
    return load_cfg(f'{__path__}check_file_templates/sh.py')


def __parse(parse, row):
    if row.startswith(parse):
        return row.strip(parse)
    return


def ___chk_fmt():
    """."""
    pass


def main(fname, chk=2):
    """Main."""
    assert chk in (-1, 1, 2, 3, 4, 5, 6, 7)
    fext = '.' + fname.split('.')[-1]
    if fext not in __checks__.keys():
        return
    else:
        cfg = __load_func(__checks__[fext])

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
            if chk == 1:
                pass
            if chk == 2:
                toret = __parse(cfg.doc_parsing, row)
                if toret:
                    cancel = True
            if (chk == 3) or (chk == -1):
                toret.append(fname)
                toret.append(__parse(cfg.doc_parsing, row))
                if toret:
                    cancel = True
            if (chk == 4) or (chk == -1):
                toret.append(__parse(cfg.globals_parsing, row))
            if (chk == 5) or (chk == -1):
                toret.append(__parse(cfg.function_parsing, row))
            if (chk == 6) or (chk == -1):
                toret.append(__parse(cfg.author_parsing, row))
            if (chk == 7) or (chk == -1):
                toret.append(__parse(cfg.version_parsing, row))
            if chk == -1:
                cancel = False
    if toret is not None:
        if len(toret) == 0:
            print(f'{FAIL}{fname} doesn\'t have parseable lines.{_RST_}')
        return strip(toret, '\n')
    else:
        print(f'{FAIL}{fname} doesn\'t have parseable lines.{_RST_}')
        return strip(toret, '\n')

# end of code
