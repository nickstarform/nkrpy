"""Reproject image."""
# flake8: noqa
# internal modules

# external modules
import numpy as np

# relative modules
from ..misc.functions import typecheck
from ..io.fits import reference

# global attributes
__all__ = ('decimal_format', 'general_format',
           'fortran_format', 'wrapper')
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)


# 158.1277,142.000001201,-0.00277777777942,339
# -21.4501305556, 41.9999987994,0.00277777777942,96
# given image header info, creates array of projects

# generating gcoord vals from known header
glon = ref(158.1277,142.000001201,-0.00277777777942,339)
glat = ref(-21.4501305556, 41.9999987994,0.00277777777942,96)
newa = np.ndarray((glon.shape[0],glat.shape[0],2),dtype=float)
for i,l in enumerate(glon):
    for j,b in enumerate(glat):
        newa[i,j] = ga2equ((l,b))

# placing vals in more appropriate names
# values  of the new header
ncrpix = tuple([int(x/2) for x in newa.shape][:-1])
ncrval = tuple(newa[ncrpix[0],ncrpix[1],:].tolist())
ncrdel = tuple(newa[ncrpix[0]+1,ncrpix[1]+1,:]-newa[ncrpix[0],ncrpix[1],:])


from astropy.io import fits
from astropy import wcs
import matplotlib.pyplot as plt

# reading in appropriate info from file
filename = '12co_allregions.mom0.fits'
hdulist  = fits.open(filename)
imheader = hdulist[0].header
imdata   = hdulist[0].data
coords   = wcs.WCS(imheader)

headervals = ('CTYPE1',
'CRVAL1',
'CDELT1',
'CRPIX1',
'CUNIT1',
'CTYPE2',
'CRVAL2',
'CDELT2',
'CRPIX2',
'CUNIT2')
# These are the header values, have to override them by hand because I am lazy
# example
# imheader['CRVAL1'] = ncrval[0]

# override the image file and write out once finished
hdulist[0].header = imheader
hdulist.writeto('newfile.reproj.fits')


#########################################################################################
# IGNORE
##############
filename = '12co_allregions.mom0.reproj.fits'
hdulist  = fits.open(filename)
imheader = hdulist[0].header
imdata   = hdulist[0].data
coords   = wcs.WCS(imheader)

imheader['CTYPE1'] = 'DEC--SIN' 
imheader['CRVAL1'] = 30.73189202208101
imheader['CDELT1'] = 0.003771434856041367
imheader['CRPIX1'] = 48
imheader['CUNIT1'] = 'deg'
imheader['CTYPE2'] = 'RA--SIN' 
imheader['CRVAL2'] = 51.39622802949072
imheader['CDELT2'] = -0.00049141541337860
imheader['CRPIX2'] = 169
imheader['CUNIT2'] = 'deg'


# imheader['CRVAL1'] = ncrval[0]

# override the image file and write out once finished
