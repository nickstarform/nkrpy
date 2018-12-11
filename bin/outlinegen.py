"""Generate the outline for the nkrpy package."""

# internal modules

# external modules

# relative modules
from nkrpy.functions import list_files_fmt

# global attributes
__all__ = ('test', 'main')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def main():
    """Main caller function."""
    with open('../outline.md','w') as f:
        for row in list_files_fmt('/home/reynolds/github/nickalaskreynolds/nkrpy',
            'build,egg,__init__,.pyc,__pycache__,.git,dist,ipynb_'):
            f.write(f'{row}\n')
        f.write('\n')
    pass


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""
    print('Generating Outline')
    main()
    print('Completed Outline')

# end of code
