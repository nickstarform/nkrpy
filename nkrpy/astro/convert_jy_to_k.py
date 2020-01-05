"""Convert a given image flux density."""

# internal modules
from argparse import ArgumentParser as ap

# internal modules

# external modules

# relative modules
from .radio_functions import convert_file

# relative modules

# global attributes
__all__ = ()
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


if __name__ == "__main__":
    """Directly Called."""
    parser = ap()
    parser.add_argument('-i', type=str, required=True)
    parser.add_argument('-jy', default=False, action='store_true', help='convert to jy') # noqa
    parser.add_argument('-k', default=False, action='store_true', help='convert to jy') # noqa
    args = parser.parse_args()

    print('Running module')
    convert_file(args.i, args.k, args.jy)
    print('Run Passed')

# end of code
