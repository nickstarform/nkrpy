"""Module loads/manipulates/saves its files."""

# import modules
from astropy.io import fits
import os


def read(fname):
    """Read in the file and neatly close."""
    header, data = None, None
    with fits.open(fname) as hdul:
        header = hdul[0].header
        data = hdul[0].data
    return header, data


def write(f, fname=None, header=None, data=None):
    """Open and read from the file."""
    if isinstance(f, str):
        # if string
        if os.path.isfile(f):
            # open and update
            with fits.open(f, mode='update') as hdul:
                if header:
                    header = hdul[0].header
                if data:
                    data = hdul[0].data
                hdul.flush()
        else:
            # write new
            hdu = fits.PrimaryHDU(data)
            if header:
                hdu.header = header
            hdu.writeto(f)
    elif isinstance(f, fits.hdu.hdulist.HDUList):
        if header:
            f[0].header = header
        if data:
            f[0].data = data
        if fname:
            f.writeto(fname)
        else:
            f.flush()
    elif isinstance(f, fits.hdu.image.PrimaryHDU):
        if header:
            f.header = header
        if data:
            f.data = data
        if fname:
            f.writeto(fname)
        else:
            f.flush()
    return True

# end of file
