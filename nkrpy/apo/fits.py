"""Module loads/manipulates/saves its files."""

# internal modules
import os
import re

# external modules
from astropy.io import fits
import numpy as np

# relative modules
from ..functions import typecheck

# global attributes
__all__ = ('read', 'write')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__version__ = 0.1


def create_header(h):
    """Create header from string."""
    if isinstance(h, tuple) or isinstance(h, list) or isinstance(h, set):
        return list(h)
    if isinstance(h, str) or isinstance(h, fits.header.Header):
        h = str(h)
        regex = "(\S*\s*)(=)"
        matches = re.finditer(regex, str(h), re.MULTILINE)
        for num, match in enumerate(matches, start=1):
            linestart = match.group()
        h = h.replace(linestart, f';;;{linestart}')
        tmp = [line for line in h.split(';;;')]
        return tmp
    else:
        return None


def read(fname):
    """Read in the file and neatly close."""
    header, data = [], []
    with fits.open(fname) as hdul:
        for h in hdul:
            if 'XTENSION' in list(map(lambda x: x.upper(), h.header.keys())):
                if h.header['XTENSION'].upper() == 'BINTABLE':
                    continue
            header.append(h.header)
            data.append(h.data.astype(float))
    return header, data


def write(f, fname=None, header=None, data=None):
    """Open and read from the file."""
    if not isinstance(header, fits.header.Header):
        header = create_header(header)
        if typecheck(header):
            while len(header) < (36 * 4 - 1):
                header.append('')  # Adds a blank card to the end
    if isinstance(f, str):
        # if string
        print('Found string')
        if isinstance(fname, str):
            f = fname
        if os.path.isfile(f):
            # open and update
            print('Updating File')
            with fits.open(f, mode='update') as hdul:
                if header is not None:
                    header = hdul[0].header
                if data is not None:
                    data = hdul[0].data
                hdul.flush()
            return
        else:
            # write new
            print('Making new file')
            if header is not None:
                hdu = fits.PrimaryHDU(data, header)
            else:
                hdu = fits.PrimaryHDU(data)
            hdu.writeto(f)

    elif isinstance(f, fits.hdu.hdulist.HDUList):
        print('Found HDUList')
        if header is not None:
            f[0].header = header
        if data is not None:
            f[0].data = data
        if fname is not None:
            f.writeto(fname)
        else:
            f.flush()
    elif isinstance(f, fits.hdu.image.PrimaryHDU):
        print('Found PrimaryHDU')
        if header is not None:
            f.header = header
        if data is not None:
            f.data = data
        if fname is not None:
            f.writeto(fname)
        else:
            f.flush()
    return True


def make_nan(filename: str):
    header, data = read(filename)
    data[:] = np.nan
    write(f'nan_{filename}', header=header, data=data)

def make_zero(filename: str):
    header, data = read(filename)
    data[:] = 0
    write(f'zero_{filename}', header=header, data=data)

def header_radec(header: dict):
    # check for hese headers CRTYPE, CRPIX, CRVAL, CDELT/CD
    size = (header['NAXIS1'], header['NAXIS2'])
    ra_pix = header['CRPIX1']
    dec_pix = header['CRPIX2']
    ra_val = header['CRVAL1']
    dec_val = header['CRVAL2']
    if 'CDELT1' in header.keys():
        ra_del = header['CDELT1']
        dec_del = header['CDELT2']
    else:
        ra_del = header['CD1_1']
        dec_del = header['CD2_2']
    ra_begin = ra_val - ra_pix * ra_del
    dec_begin = dec_val - dec_pix * dec_del
    rapix = np.arange(0, size[0], 1) * ra_del + ra_begin
    decpix = np.arange(0, size[1], 1) * dec_del + dec_begin
    return rapix, decpix



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
