"""Generate the outline for the nkrpy package."""

# internal modules
import os
from datetime import datetime

# external modules

# relative modules
from nkrpy.files import list_files_fmt
from nkrpy.__info__ import *

# global attributes
__all__ = ('main',)
__doc__ = """This file fully explores all directories of the module `nkrpy`."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def main():
    """Main caller function."""
    fname = './outline.rst'
    title = 'Outline'
    spacing = ''.join(["=" for x in range(len(title) + 2)])
    with open(fname, 'w') as f:
        f.write(f'{spacing}\n')
        f.write(f'{title}\n')
        f.write(f'{spacing}\n\n')
        f.write(f':Web: `{package_web}`_\n')
        f.write(f':Author: `{author}`_ <{email}>\n')
        f.write(f':Author Web: `{author_web}`_\n')
        f.write(f':Date: {str(datetime.now())}\n')
        f.write(f':Description: {__doc__}\n')
        f.write(f':Desc. Cont...: This file is auto-generated\n\n')
        f.write(f'.. _`{author}`: mailto:{email}\n')
        f.write(f'.. _`{author_web}`: {author_web}\n')
        f.write(f'.. _`{package_web}`: {package_web}\n\n')
    with open(fname, 'a') as f:
        for row in \
            list_files_fmt('/home/reynolds/github/nickalaskreynolds/nkrpy',
                           'build,egg,__init__,.pyc,__pycache__,' +
                           '.git,dist,ipynb_',
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
