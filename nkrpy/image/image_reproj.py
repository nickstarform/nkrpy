
from math import *
import numpy as np

def ga2equ(ga):
    """
    Convert Galactic to Equatorial coordinates (J2000.0)
    (use at own risk)
    
    Input: [l,b] in decimal degrees
    Returns: [ra,dec] in decimal degrees
    
    Source: 
    - Book: "Practical astronomy with your calculator" (Peter Duffett-Smith)
    - Wikipedia "Galactic coordinates"
    
    Tests (examples given on the Wikipedia page):
    >>> ga2equ([0.0, 0.0]).round(3)
    array([ 266.405,  -28.936])
    >>> ga2equ([359.9443056, -0.0461944444]).round(3)
    array([ 266.417,  -29.008])
    """
    l,b = map(radians,ga)
    # North galactic pole (J2000) -- according to Wikipedia
    pole_ra = radians(192.859508)
    pole_dec = radians(27.128336)
    posangle = radians(122.932-90.0)
    # North galactic pole (B1950)
    #pole_ra = radians(192.25)
    #pole_dec = radians(27.4)
    #posangle = radians(123.0-90.0)
    ra = atan2( (cos(b)*cos(l-posangle)), (sin(b)*cos(pole_dec) - cos(b)*sin(pole_dec)*sin(l-posangle)) ) + pole_ra
    print(ra)
    dec = asin( cos(b)*cos(pole_dec)*sin(l-posangle) + sin(b)*sin(pole_dec) )
    return (degrees(ra), degrees(dec))

# 158.1277,142.000001201,-0.00277777777942,339
# -21.4501305556, 41.9999987994,0.00277777777942,96
# given image header info, creates array of projects
def ref(crval,crpix,cdelt,num):
    a = np.arange(1,num+1,1,dtype=float)
    a[int(crpix)] = crval
    for i,x in enumerate(a):
        delpix = i - int(crpix)
        delpix_delta = delpix * cdelt
        a[i] = crval + delpix_delta
    return a

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

headervals = ('CTYPE1'                                                           ,
'CRVAL1'                                             ,
'CDELT1'                                                 ,
'CRPIX1'                                                ,
'CUNIT1'                                                            ,
'CTYPE2'                                                           ,
'CRVAL2'                                                ,
'CDELT2'                                                ,
'CRPIX2'   ,                                              
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
