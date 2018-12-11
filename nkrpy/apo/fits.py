"""Module loads/manipulates/saves its files."""

# internal modules
import os

# external modules
from astropy.io import fits

# relative modules

# global attributes
__all__ = ('read', 'write')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def header():
    """Create header."""
    pass


def read(fname):
    """Read in the file and neatly close."""
    header, data = None, None
    with fits.open(fname) as hdul:
        header = hdul[0].header
        data = hdul[0].data.astype(float)
    return header, data


def write(f, fname=None, header=None, data=None):
    """Open and read from the file."""
    if isinstance(f, str):
        # if string
        if os.path.isfile(f):
            # open and update
            with fits.open(f, mode='update') as hdul:
                if header is not None:
                    header = hdul[0].header
                if data is not None:
                    data = hdul[0].data
                hdul.flush()
        else:
            # write new
            hdu = fits.PrimaryHDU(data)
            if header is not None:
                hdu.header = header
            hdu.writeto(f)
    elif isinstance(f, fits.hdu.hdulist.HDUList):
        if header is not None:
            f[0].header = header
        if data is not None:
            f[0].data = data
        if fname is not None:
            f.writeto(fname)
        else:
            f.flush()
    elif isinstance(f, fits.hdu.image.PrimaryHDU):
        if header is not None:
            f.header = header
        if data is not None:
            f.data = data
        if fname is not None:
            f.writeto(fname)
        else:
            f.flush()
    return True


def main():
    """Main caller function."""
    pass


def test():
    """Testing function for module."""
    pass


if __name__ == "__main__":
    """Directly Called."""

    print('Testing module')
    test()
    print('Test Passed')

# end of code
