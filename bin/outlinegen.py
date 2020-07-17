"""Generate the outline for the nkrpy package.

This file fully explores all directories of
    the module `nkrpy` and generates the outline.
    It is a little convoluted and if nkrpy already
    exists on the system, will favour those commands
    over the commands listed here.
"""

# internal modules
import os
from datetime import datetime
import sys
from collections.abc import Iterable
import importlib
# external modules

# relative modules

# global attributes
__all__ = ('main',)
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
if __path__ == '/bin/':
    __path__ = os.getcwd() + '/bin'
else:
    __path__ = __path__ + '/../'


# try importing settings and nkrpy if it exists
sys.path.append(os.path.dirname(f'{__path__}'))
import setup # noqa
settings = setup.settings
nkrpy = importlib.util.find_spec("nkrpy.io", package="files")
found = nkrpy is not None
if found:
    from nkrpy.io.files import list_files_fmt as nkrpy


# incase nkrpy doesn't exist, declare basic functions
def typecheck(obj):
    """Check if object is iterable (array, list, tuple) and not string."""
    return not isinstance(obj, str) and isinstance(obj, Iterable)


def list_comp(base, comp):
    """Compare 2 lists, make sure purely unique (True)."""
    l1 = set(comp)
    l2 = set(base)
    if (l1 - l2) != l1:
        return False
    if (l2 - l1) != l2:
        return False
    return True


def addspace(arbin, spacing='auto'):
    if typecheck(arbin):
        if str(spacing).lower() == 'auto':
            spacing = max([len(x) for x in map(str, arbin)]) + 1
            return [_add(x, spacing) for x in arbin]
        elif isinstance(spacing, int):
            return [_add(x, spacing) for x in arbin]
    else:
        arbin = str(arbin)
        if str(spacing).lower() == 'auto':
            spacing = len(arbin) + 1
            return _add(arbin, spacing)
        elif isinstance(spacing, int):
            return _add(arbin, spacing)
    raise TypeError(f'Either input: {arbin} or spacing:' +
                    f'{spacing} are of incorrect types. NO OBJECTS')


def _add(sstring, spacing=20):
    """Regular spacing for column formatting."""
    sstring = str(sstring)
    while True:
        if len(sstring) >= spacing:
            sstring = sstring[:-1]
        elif len(sstring) < (spacing - 1):
            sstring = sstring + ' '
        else:
            break
    return sstring + ' '


def _strip(array, var=''):
    """Kind of a quick wrapper for stripping lists."""
    if array is None:
        return
    elif isinstance(array, str):
        return array.strip(var)
    _t = []
    for x in array:
        x = x.strip(' ').strip('\n')
        if var:
            x = x.strip(var)
        if x != '':
            _t.append(x)
    return tuple(_t)


def list_files_fmt(startpath, ignore='',
                   formatter=['  ', '| ', 1, '|--', ''],
                   pad=False, append=False, header_wrap=None, docs=False):
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
    if found:  # if nkrpy exists
        return nkrpy(startpath, ignore, formatter,
                     pad, append, header_wrap, docs)
    OKBLUE = '\033[94m'
    _RST_ = '\033[0m'  # resets color and format
    # spacing, level iterator, same level iterator num, level terminator
    s, a, b, c, e = formatter
    full = []
    ignore = _strip(ignore.split(','))
    for (root, _, files) in os.walk(startpath):
        level = root.replace(startpath, '').count(os.sep)
        indent = s + a * b * level + c
        base = os.path.basename(root)

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
            len_lvl = max(list(map(lambda x: len(x), files))) + 1
            for i, f in enumerate(files):
                _f_ck = '/'.join(root) + f'/{f}'
                print(f'{OKBLUE}{i + 1}/{len(files)}: {_f_ck}{_RST_}')
                if pad is True:
                    f = addspace(f, len_lvl)
                if isinstance(append, str):
                    f += append
                if list_comp(_f_ck.split('/'), ignore):
                    full.append('{}{}{}'.format(subindent, f, e))

    return tuple(full)


# main caller
def main():
    """Main."""
    fname = './outline.rst'
    title = 'Outline'
    spacing = ''.join(["=" for x in range(len(title) + 2)])
    with open(fname, 'w') as f:
        f.write(f'{spacing}\n')
        f.write(f'{title}\n')
        f.write(f'{spacing}\n\n')
        f.write(f':Web: `{settings["name"]}`_\n')
        f.write(f':Author: `{settings["author"]}`_ ' +
                f'{settings["author_email"]}\n')
        f.write(f':Date: {str(datetime.now())}\n')
        f.write(f':Description: {settings["description"]}\n')
        f.write(f':Desc. Cont...: This file is auto-generated from ' +
                f'bin/{__filename__}.py\n\n')
        f.write(f'.. _`{settings["author"]}`: ' +
                f'mailto:{settings["author_email"]}\n')
        f.write(f'.. _`{settings["name"]}`: {settings["url"]}\n\n')

    with open(fname, 'a') as f:
        for row in \
            list_files_fmt('nkrpy',
                           'build,egg,__info__.py,__init__.py,' +
                           '.pyc,__pycache__,' +
                           '.git,dist,ipynb_,orbital,' +
                           '__init_generator__.py,__main__.py, .so',
                           formatter=['', '  ', 1, '* ', '\n'],
                           header_wrap=['both', '**'],
                           pad=True, append='<--', docs=True):
            f.write(f'{row}\n')
        f.write('\n\n')
    for dt in ['pdf', 'html5.py']:
        dto = dt
        if dt.endswith('.py'):
            dto = dt.strip('.py')
        else:
            dto = dt + ' -s .rst_pdf.json'
        os.system(f'rst2{dt} outline.rst outline.{dto}')
    pass


if __name__ == "__main__":
    """Directly Called."""
    print('Generating Outline')
    main()
    print('Completed Outline')

# end of code
