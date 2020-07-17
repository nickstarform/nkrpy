
import astropy.units as u
import astropy.constants as co
import scipy as _sp
import numpy as np
import astropy.io.fits as fits
import sys
from argparse import ArgumentParser


########################################################################
def deg2rad(x):
    return x*np.pi/180.0
    
# DE-PROJECT i.e. ROTATE, INCLINE

def deproject(uv, PA, inc):
    """
    Rotate and deproject individual visibility coordinates.
    From Hughes et al. (2007) - "AN INNER HOLE IN THE DISK AROUND 
    TW HYDRAE RESOLVED IN 7 mm DUST EMISSION".
    """
    R = ( (uv**2).sum(axis=0) )**0.5
    #~ phi = _sp.arctan(uv[1]/uv[0] - deg2rad(PA))
    phi = _sp.arctan2(uv[1],uv[0]) - deg2rad(PA)
    #~ phi = _sp.arctan2( (uv[1] - deg2rad(PA) * uv[0]) , uv[0])
    newu = R * _sp.cos(phi) * _sp.cos( deg2rad(inc) )
    newv = R * _sp.sin(phi)
    newuv = _sp.array([newu, newv])
    ruv = (newuv**2).sum(axis=0)**.5
    return newuv, ruv

def rotate_field(uv, PA, U_RA_align = True):
    """
    Rotates a coordinate system (UV plane) by PA
    degrees.
    uv : 2-D array with uv[0] U and uv[1] coordinated
    PA : Position Angle, in degrees
    U_RA_align : for ALMA and PdBI the U-axis and RA are aligned
                 and thus one form of the equation must be used
                 While for SMA/CARMA (USA, meh), they are not aligned
                 and thus some sign changes have to impliemented from
                 that presented in Berger & Segransan (2007)
    
    """
    direction =  [-1, 1][int(U_RA_align)]
    u_new = uv[0] * _sp.cos( deg2rad(PA) ) + direction * uv[1] * _sp.sin( deg2rad(PA) )
    v_new = -1 * direction * uv[0] * _sp.sin( deg2rad(PA) ) + uv[1] * _sp.cos( deg2rad(PA) )
    return u_new, v_new
    
def incline(uv, inc):
    #~ ruv = ( uv[0]**2 + (uv[1] * _sp.cos(deg2rad(inc)) )**2  )**.5 
    # the PA goes from North to East in the image plane, and the 
    # Major axis is flipped 90 degrees going from
    # image to UV plane (Major in image in minor in UV)
    ruv = ( uv[0]**2 * _sp.cos(deg2rad(inc))**2 + uv[1]**2  )**.5 
    return ruv


if __name__ == '__main__':

    description = 'Deprojects visibility data'\
                  'can also handle comma separate values'

    in_help = 'name of the file to deproject (FITS FORMAT) include extension'
    out_help = 'name of the output file include extension. if unspecified, the file will be named in the format: INPUT_FILE_DEPROJECT.fits'
    pa_help = 'Position angle of the source'
    inc_help = 'Inclination of the source'

    # Initialize instance of an argument parser
    parser = ArgumentParser(description=description)

    # Add required arguments
    parser.add_argument('--input', help=in_help, type=str,dest='infil')

    parser.add_argument('-p', '--pa', type=str, help=pa_help,dest='posa')
    parser.add_argument('-i', '--inc', type=str, help=inc_help,dest='inclin')


    # Add optional arguments, with given default values if user gives no args
    parser.add_argument('-o', '--output', type=str, help=out_help,dest='outfil')

    # Get the arguments
    args = parser.parse_args()


    # define input
    print('Parsing input.')

    while True:
        try:
            _TEMP1_=args.infil
            _TEMPARRAY1_=_TEMP1_.split(',')
            _TEMP2_=args.posa
            _TEMPARRAY2_=_TEMP2_.split(',')
            _TEMP3_=args.inexpclin
            _TEMPARRAY3_=_TEMP3_.split(',')
            _TEMP4_=args.outfil
            _TEMPARRAY4_=_TEMP4_.split(',')
            if (_TEMPARRAY1_) and (_TEMPARRAY2_) and (_TEMPARRAY3_):
                print('Everything read in properly...Continuing...')
                break
            else:
                print('Please CTRL+C to stop code and fix input error. Use --help to view inputs.')
        except ValueError:
            print('Error on inputs. Try again.')
            sys.exit()
        except AttributeError:
            if not _TEMP4_:
                print('No output supplied. Continuing...')
            if (_TEMPARRAY1_) and (_TEMPARRAY2_) and (_TEMPARRAY3_):
                print('Everything read in properly...Continuing...')
                break
            else:
                print('Use --help to view inputs.')
                sys.exit()


    for _NUM_,_INPUT_ in enumerate(_TEMPARRAY1_):
        infile=_TEMPARRAY1_[_NUM_]
        pa=float(_TEMPARRAY2_[_NUM_])
        inc=float(_TEMPARRAY3_[_NUM_])
        if not _TEMP4_:
            outfile=_TEMPARRAY1_[_NUM_].strip('.txt').strip('.FIT').strip('.FITS').strip('.fit').strip('.fits').strip('.img').strip('.IMAGE').strip('.image')
        else:
            outfile=_TEMPARRAY4_[_NUM_]+'.deproj.fits'

        # get the data and restfreq
        print('Reading input UV-FITS data: ' + infile)
        hdu = fits.open(infile)
        #restfreq = hdu[0].header['RESTFRQ']*u.Hz
        try:
            uu = hdu[0].data['UU']
            vv = hdu[0].data['VV']
        except IndexError:
            print('Export the MS to uv fits .')
            continue
        # deproject the coordinates
        print('Deprojecting data.')
        uuvv_new, ruv_new = deproject(np.array([uu,vv]), pa, inc)
        hdu[0].data['UU'] = uuvv_new[0]
        hdu[0].data['VV'] = uuvv_new[1]
        print('Saving output file: ' + outfile)
        hdu.writeto(outfile)