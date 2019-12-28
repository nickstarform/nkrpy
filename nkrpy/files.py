"""This code finds the jacobi constant from given files."""

# standard modules
import os
import shutil

# external modules

# relative Modules
from .colours import _RST_, OKBLUE
from .functions import typecheck, list_comp, addspace, _strip
from . import check_file

# global attributes
__all__ = ('copytree', 'list_files',
           'list_files_fmt', 'freplace')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


def freplace(filein, olddata, newdata):
    """Replace a string in a file."""
    with open(filein, 'r') as f:
        filedata = f.read()

    newdata = filedata.replace(olddata, newdata)

    with open(filein, 'w') as f:
        f.write(newdata)
    pass


def copytree(src, dst, symlinks=False, ignore=None):
    """Better wrapper for copying tree directives."""
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            shutil.copytree(s, d, symlinks, ignore)
        else:
            shutil.copy2(s, d)


def list_files(dir):
    """List all the files within a directory."""
    r = []
    subdirs = [x[0] for x in os.walk(dir)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if (len(files) > 0):
            for file in files:
                r.append(subdir + "/" + file)
    return r


def list_files_fmt(startpath, ignore='',
                   formatter=['  ', '| ', 1, '|--', ''],
                   pad=False, append=False, docs=False, header_wrap=None):
    """Intelligent directory stepper + ascii plotter.

    Will walk through directories starting at startpath
    ignore is a csv string 'ignore1, ignore2...' that will ignore any
        file or directory with same name
    Formatter will format the output in:
        [starting string for all lines,
         the iterating level parameter,
         the number of iterations for the level parameter per level,
         the final string to denote the termination at file/directory]
    header_wrap: 2 element iterable (a/b/both, string)
    pad: will pad each ending element by the length
    append: append string at end of every ending element
    docs: if true, will try to generate docs for objects too
    example:
    |--/
    | |--CONTRIBUTING.md
    | |--.gitignore
    | |--LICENSE
    | |--CODE_OF_CONDUCT.md
    | |--README.md
    | |--PULL_REQUEST_TEMPLATE.md
    | |--refs/
    | | |--heads/
    | | | |--devel
    """
    # spacing, level iterator, same level iterator num, level terminator
    s, a, b, c, e = formatter
    full = []
    ignore = _strip(ignore.split(','))
    for root, firs, files in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = s + a * b * level + c
        base = os.path.basename(root)
        _f = None

        root = root.split('/')
        if list_comp(root, ignore):
            dir_base = f'{base}/'
            if typecheck(header_wrap):
                if header_wrap[0] == 'both' or header_wrap[0] == 'b':
                    dir_base = f'{header_wrap[1]}{dir_base}'
                if header_wrap[0] == 'both' or header_wrap[0] == 'a':
                    dir_base = f'{dir_base}{header_wrap[1]}'
            full.append(f'{indent}{dir_base}{e}')
            subindent = s + a * b * (level + 1) + c
            if len(files) == 0:
                continue
            len_lvl = max(list(map((lambda x: len(x)), files))) + 1
            for i, f in enumerate(files):
                _f_ck = '/'.join(root) + f'/{f}'
                print(f'{OKBLUE}{i + 1}/{len(files)}: {_f_ck}{_RST_}')
                if pad is True:
                    f = addspace(f, len_lvl)
                if isinstance(append, str):
                    f += append
                if docs is True:
                    _f = check_file.main(_f_ck, 2)
                    if _f is not None:
                        f += _f
                if list_comp(_f_ck.split('/'), ignore):
                    full.append('{}{}{}'.format(subindent, f, e))
    return tuple(full)

# end of code

# end of file
